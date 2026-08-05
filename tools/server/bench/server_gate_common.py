#!/usr/bin/env python3
"""Shared process/HTTP discipline for self-booting server gate drivers."""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request


def request(base, path, body=None, timeout=600):
	headers = {"Content-Type": "application/json"} if body is not None else {}
	data = json.dumps(body).encode("utf-8") if body is not None else None
	req = urllib.request.Request(
		base + path, data=data, headers=headers,
		method="POST" if data is not None else "GET")
	try:
		with urllib.request.urlopen(req, timeout=timeout) as response:
			payload = response.read()
			return (response.status,
				json.loads(payload) if payload else None,
				dict(response.headers.items()))
	except urllib.error.HTTPError as error:
		payload = error.read()
		try:
			decoded = json.loads(payload) if payload else None
		except Exception:
			decoded = {"raw": payload.decode(errors="replace")}
		return error.code, decoded, dict(error.headers.items())


def header(headers, name):
	for key, value in headers.items():
		if key.lower() == name.lower():
			return value
	return None


def write_bench_stamp(workdir, gate, model, server_bin):
	with open(os.path.join(workdir, "BENCH_STAMP"), "w") as handle:
		handle.write(json.dumps({
			"gate": gate,
			"created_unix": int(time.time()),
			"model": os.path.abspath(model),
			"server_bin": os.path.abspath(server_bin),
		}, sort_keys=True) + "\n")


class ManagedServerArm:
	"""Server process whose context-manager terminal always drains the child."""

	def __init__(self, name, base, log_path, command):
		self.name = name
		self.base = base
		self.log_path = log_path
		self.log = open(log_path, "w")
		self.proc = subprocess.Popen(
			command, stdout=self.log, stderr=subprocess.STDOUT)

	def wait_healthy(self, timeout=360):
		deadline = time.time() + timeout
		while time.time() < deadline:
			try:
				if request(self.base, "/health", timeout=5)[0] == 200:
					return
			except Exception:
				pass
			if self.proc.poll() is not None:
				raise RuntimeError(
					f"arm {self.name} exited early; see {self.log_path}")
			time.sleep(1)
		raise RuntimeError(f"arm {self.name} never became healthy")

	def stop(self):
		if self.proc.poll() is None:
			self.proc.terminate()
			try:
				self.proc.wait(timeout=30)
			except subprocess.TimeoutExpired:
				self.proc.kill()
				self.proc.wait(timeout=30)
		if not self.log.closed:
			self.log.close()

	def __enter__(self):
		self.wait_healthy()
		return self

	def __exit__(self, exc_type, exc, traceback):
		self.stop()
		return False
