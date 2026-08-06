#!/usr/bin/env python3
"""Recording proxy between an agentic harness and llama-server.

Sits in front of an unmodified llama-server and writes one JSONL record per
request: arrival offset, the verbatim request body, wire timings (TTFT and
total), and whatever the server reported about prompt reuse. The upstream
server is never modified, so the same instrument scores a stock build and a
fork build identically.

	./capture-proxy.py --listen 8099 --upstream 127.0.0.1:8080 \
		--trace /path/to/private/hermes-website.jsonl

Point the harness at the listen port and work normally. Traces contain the
full project content: keep them out of the public repo.
"""

import argparse
import hashlib
import http.client
import json
import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOP_BY_HOP = {
	"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
	"te", "trailers", "transfer-encoding", "upgrade",
}
SECRET_HEADERS = {"authorization", "x-api-key", "api-key", "cookie"}
BLOCK = 4096
TEXT_CAP = 2 << 20


def now_ms():
	return time.monotonic() * 1000.0


def sha16(text):
	return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def request_fingerprint(body):
	"""Fields the family/foreground classifier needs, without the content."""
	out = {
		"system_prefix_sha": None,
		"system_chars": 0,
		"prompt_chars": 0,
		"n_messages": 0,
		"n_predict": None,
		"tools": 0,
	}
	if not isinstance(body, dict):
		return out
	out["n_predict"] = body.get("n_predict", body.get("max_tokens"))
	tools = body.get("tools")
	if isinstance(tools, list):
		out["tools"] = len(tools)
	messages = body.get("messages")
	if isinstance(messages, list):
		out["n_messages"] = len(messages)
		system = ""
		for message in messages:
			if not isinstance(message, dict):
				continue
			if message.get("role") != "system":
				break
			content = message.get("content")
			if isinstance(content, str):
				system += content
		out["system_chars"] = len(system)
		if system:
			# Prefix hashes at several depths: the classifier compares how far
			# two requests agree, which is the tell that separates a tooled
			# sub-agent from a tight background call.
			out["system_prefix_sha"] = sha16(system[:2048])
			out["system_prefix_shas"] = {
				str(n): sha16(system[:n]) for n in (256, 512, 1024, 2048, 4096)
				if len(system) >= min(n, len(system))
			}
		joined = "".join(
			m.get("content") or "" for m in messages
			if isinstance(m, dict) and isinstance(m.get("content"), str))
		out["prompt_chars"] = len(joined)
	prompt = body.get("prompt")
	if isinstance(prompt, str):
		out["prompt_chars"] = len(prompt)
		out["system_prefix_sha"] = sha16(prompt[:2048])
	return out


def extract_server_metrics(payload):
	"""Pull whatever the server volunteered about reuse. Shape varies by
	endpoint and build; record what exists, decide at scoring time."""
	found = {}
	if not isinstance(payload, dict):
		return found
	timings = payload.get("timings")
	if isinstance(timings, dict):
		found["timings"] = timings
	usage = payload.get("usage")
	if isinstance(usage, dict):
		found["usage"] = usage
	for key in ("tokens_evaluated", "tokens_cached", "tokens_predicted",
			"id_slot", "truncated", "stop_type"):
		if key in payload:
			found[key] = payload[key]
	return found


def sse_payloads(buffer):
	"""Yield parsed JSON objects from accumulated SSE text."""
	for line in buffer.splitlines():
		line = line.strip()
		if not line.startswith("data:"):
			continue
		data = line[5:].strip()
		if not data or data == "[DONE]":
			continue
		try:
			yield json.loads(data)
		except ValueError:
			continue


def sse_has_content(payload):
	if not isinstance(payload, dict):
		return False
	if payload.get("content"):
		return True
	choices = payload.get("choices")
	if isinstance(choices, list):
		for choice in choices:
			if not isinstance(choice, dict):
				continue
			delta = choice.get("delta")
			if isinstance(delta, dict) and delta.get("content"):
				return True
			if choice.get("text"):
				return True
	return False


class TraceWriter:
	def __init__(self, path, tag):
		self.path = path
		self.lock = threading.Lock()
		self.seq = 0
		self.start = now_ms()
		os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
		self.handle = open(path, "a", buffering=1)
		self.write_header(tag)

	def write_header(self, tag):
		header = {
			"record": "capture_header",
			"schema": "capture_trace/v1",
			"tag": tag,
			"started_unix": time.time(),
		}
		with self.lock:
			self.handle.write(json.dumps(header) + "\n")

	def next_seq(self):
		with self.lock:
			seq = self.seq
			self.seq += 1
			return seq

	def emit(self, record):
		line = json.dumps(record)
		with self.lock:
			self.handle.write(line + "\n")


class ProxyHandler(BaseHTTPRequestHandler):
	protocol_version = "HTTP/1.1"

	def log_message(self, fmt, *args):
		pass

	def do_GET(self):
		self.relay("GET")

	def do_POST(self):
		self.relay("POST")

	def do_DELETE(self):
		self.relay("DELETE")

	def forward_headers(self):
		headers = {}
		for key, value in self.headers.items():
			if key.lower() in HOP_BY_HOP:
				continue
			headers[key] = value
		return headers

	def relay(self, method):
		trace = self.server.trace
		length = int(self.headers.get("Content-Length") or 0)
		raw_body = self.rfile.read(length) if length else b""
		seq = trace.next_seq()
		t_arrival = now_ms()

		parsed_body = None
		if raw_body:
			try:
				parsed_body = json.loads(raw_body)
			except ValueError:
				parsed_body = None

		record = {
			"record": "request",
			"seq": seq,
			"t_arrival_ms": round(t_arrival - trace.start, 3),
			"method": method,
			"path": self.path,
			"request_bytes": len(raw_body),
			"stream": bool(isinstance(parsed_body, dict)
				and parsed_body.get("stream")),
			"fingerprint": request_fingerprint(parsed_body),
		}
		if parsed_body is not None:
			record["body"] = parsed_body
		elif raw_body:
			record["body_raw_b64_omitted"] = True

		try:
			conn = http.client.HTTPConnection(
				self.server.upstream_host, self.server.upstream_port,
				timeout=self.server.timeout_s)
			conn.request(method, self.path, body=raw_body,
				headers=self.forward_headers())
			response = conn.getresponse()
		except Exception as error:
			record["status"] = 599
			record["error"] = str(error)
			record["total_ms"] = round(now_ms() - t_arrival, 3)
			trace.emit(record)
			self.send_error(502, "upstream unavailable")
			return

		record["status"] = response.status
		content_length = response.getheader("Content-Length")
		streaming = content_length is None

		self.send_response(response.status)
		for key, value in response.getheaders():
			if key.lower() in HOP_BY_HOP or key.lower() == "content-length":
				continue
			self.send_header(key, value)
		if streaming:
			self.send_header("Transfer-Encoding", "chunked")
		else:
			self.send_header("Content-Length", content_length)
		self.end_headers()

		t_first_byte = None
		t_first_token = None
		received = 0
		text_buffer = ""
		final_payload = None

		try:
			while True:
				chunk = response.read(BLOCK)
				if not chunk:
					break
				if t_first_byte is None:
					t_first_byte = now_ms()
				received += len(chunk)
				if streaming:
					self.wfile.write(b"%x\r\n" % len(chunk))
					self.wfile.write(chunk)
					self.wfile.write(b"\r\n")
					self.wfile.flush()
					if len(text_buffer) < TEXT_CAP:
						text_buffer += chunk.decode("utf-8", "replace")
					if t_first_token is None:
						for payload in sse_payloads(text_buffer):
							if sse_has_content(payload):
								t_first_token = now_ms()
								break
				else:
					self.wfile.write(chunk)
					# Non-streaming bodies carry timings/usage in the single
					# payload; keep them (capped) so both response shapes score.
					if len(text_buffer) < TEXT_CAP:
						text_buffer += chunk.decode("utf-8", "replace")
			if streaming:
				self.wfile.write(b"0\r\n\r\n")
				self.wfile.flush()
		except Exception as error:
			record["stream_error"] = str(error)
		finally:
			conn.close()

		t_end = now_ms()
		record["response_bytes"] = received
		record["ttfb_ms"] = (round(t_first_byte - t_arrival, 3)
			if t_first_byte else None)
		record["ttft_ms"] = (round(t_first_token - t_arrival, 3)
			if t_first_token else record["ttfb_ms"])
		record["total_ms"] = round(t_end - t_arrival, 3)

		if streaming:
			for payload in sse_payloads(text_buffer):
				metrics = extract_server_metrics(payload)
				if metrics:
					final_payload = metrics
		else:
			try:
				final_payload = extract_server_metrics(
					json.loads(text_buffer or "{}"))
			except ValueError:
				final_payload = None
		if final_payload:
			record["server"] = final_payload

		trace.emit(record)


class ProxyServer(ThreadingHTTPServer):
	daemon_threads = True
	allow_reuse_address = True


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--listen", type=int, required=True)
	parser.add_argument("--upstream", required=True,
		help="host:port of the unmodified llama-server")
	parser.add_argument("--trace", required=True,
		help="JSONL output path (keep private: contains project content)")
	parser.add_argument("--tag", default="capture")
	parser.add_argument("--timeout", type=float, default=1800.0)
	args = parser.parse_args()

	host, _, port = args.upstream.rpartition(":")
	if not host or not port.isdigit():
		parser.error("--upstream must be host:port")

	trace = TraceWriter(args.trace, args.tag)
	server = ProxyServer(("127.0.0.1", args.listen), ProxyHandler)
	server.trace = trace
	server.upstream_host = host
	server.upstream_port = int(port)
	server.timeout_s = args.timeout

	print(f"CAPTURE_PROXY listening 127.0.0.1:{args.listen} "
		f"-> {args.upstream}, trace {args.trace}")
	sys.stdout.flush()
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
	finally:
		server.server_close()
		print(f"CAPTURE_PROXY recorded {trace.seq} requests")


if __name__ == "__main__":
	main()
