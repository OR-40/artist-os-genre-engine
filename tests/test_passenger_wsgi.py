import io
import json
import unittest
from wsgiref.util import setup_testing_defaults

from passenger_wsgi import application


class PassengerAdapterTests(unittest.TestCase):
    def test_wsgi_adapter_serves_health_route(self):
        environ = {}
        setup_testing_defaults(environ)
        environ.update({
            "REQUEST_METHOD": "GET",
            "PATH_INFO": "/health",
            "QUERY_STRING": "",
            "CONTENT_LENGTH": "0",
            "wsgi.input": io.BytesIO(b""),
        })
        captured = {}

        def start_response(status, headers, exc_info=None):
            captured["status"] = status
            captured["headers"] = dict(headers)

        body = b"".join(application(environ, start_response))
        self.assertEqual(captured["status"], "200 OK")
        self.assertIn("application/json", captured["headers"].get("content-type", ""))
        payload = json.loads(body.decode("utf-8"))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["execution_device"], "CPU")


if __name__ == "__main__":
    unittest.main()
