"""Shared wire parsing for the capstone capture/replay tools.

Response-body retention is deliberately bounded, but claim evidence is not allowed to
depend on that retention window.  The incremental SSE reader consumes every complete
data line and keeps only the unfinished line between chunks.
"""

import codecs
import json


SERVER_METRIC_FIELDS = (
	"tokens_evaluated", "tokens_cached", "tokens_predicted", "id_slot",
	"truncated", "stop_type",
)


def extract_server_metrics(payload):
	"""Return only server-owned timing/reuse/context fields from one payload."""
	found = {}
	if not isinstance(payload, dict):
		return found
	for key in ("timings", "usage"):
		value = payload.get(key)
		if isinstance(value, dict):
			found[key] = value
	for key in SERVER_METRIC_FIELDS:
		if key in payload:
			found[key] = payload[key]
	return found


def merge_server_metrics(target, payload):
	"""Merge fields observed across streaming payloads; later values are final."""
	target.update(extract_server_metrics(payload))
	return target


def sse_payloads(text):
	for line in text.splitlines():
		line = line.strip()
		if not line.startswith("data:"):
			continue
		data = line[5:].strip()
		if not data or data == "[DONE]":
			continue
		try:
			payload = json.loads(data)
		except ValueError:
			continue
		if isinstance(payload, dict):
			yield payload


class SSEPayloadAccumulator:
	"""Incrementally parse one-line JSON SSE payloads with O(max-line) storage."""

	def __init__(self):
		self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
		self._pending = ""

	@staticmethod
	def _payload(line):
		line = line.strip()
		if not line.startswith("data:"):
			return None
		data = line[5:].strip()
		if not data or data == "[DONE]":
			return None
		try:
			payload = json.loads(data)
		except ValueError:
			return None
		return payload if isinstance(payload, dict) else None

	def feed(self, chunk, final=False):
		text = self._decoder.decode(chunk, final=final)
		combined = self._pending + text
		lines = combined.splitlines(keepends=True)
		self._pending = ""
		if lines and not lines[-1].endswith(("\n", "\r")):
			self._pending = lines.pop()
		out = []
		for line in lines:
			payload = self._payload(line)
			if payload is not None:
				out.append(payload)
		if final and self._pending:
			payload = self._payload(self._pending)
			self._pending = ""
			if payload is not None:
				out.append(payload)
		return out
