"""Exercise rejected-request draining through the ASGI receive boundary."""

import time
import unittest

from fastapi import HTTPException, Request

from serve import MAX_BODY_BYTES, REJECTED_BODY_DRAIN_TIMEOUT_SECONDS, _read_json


class EarlyErrorBodyDrainTests(unittest.IsolatedAsyncioTestCase):
    async def reject(self, content_length, chunks, extra_headers=()):
        received = []
        messages = iter(chunks)

        async def receive():
            chunk, more = next(messages)
            received.append(len(chunk))
            return {"type": "http.request", "body": chunk, "more_body": more}

        headers = [(b"content-type", b"text/plain"), (b"content-length", str(content_length).encode())]
        request = Request({"type": "http", "headers": headers + list(extra_headers)}, receive=receive)
        started = time.monotonic()
        with self.assertRaises(HTTPException) as error:
            await _read_json(request)
        self.assertEqual(error.exception.status_code, 415)
        self.assertLess(time.monotonic() - started, REJECTED_BODY_DRAIN_TIMEOUT_SECONDS)
        self.assertLessEqual(sum(received), MAX_BODY_BYTES)
        return received

    async def test_oversized_single_chunk_is_rejected_before_receive(self):
        body = bytes(9 * 1024 * 1024)
        self.assertEqual(await self.reject(len(body), [(body, False)]), [])

    async def test_exactly_at_the_limit_drains_the_framed_body(self):
        body = bytes(MAX_BODY_BYTES - 1)
        self.assertEqual(await self.reject(MAX_BODY_BYTES, [(body, True), (b"x", False)]),
                         [MAX_BODY_BYTES - 1, 1])

    async def test_transfer_encoding_disables_content_length_budget(self):
        body = bytes(9 * 1024 * 1024)
        self.assertEqual(await self.reject(1, [(body, False)], [(b"transfer-encoding", b"chunked")]), [])


if __name__ == "__main__":
    unittest.main()
