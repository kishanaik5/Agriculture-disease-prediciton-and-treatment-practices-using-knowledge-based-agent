import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from SharedBackend.managers.pwdManager import UserPasswordManager
from SharedBackend.middlewares.AuthMiddleware import JWTAuthMiddleware


class TestJWTAuthMiddleware(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.pwd_manager = UserPasswordManager()
        self.app.add_middleware(JWTAuthMiddleware, pwd_manager=self.pwd_manager)

        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        self.client = TestClient(self.app)

    def test_jwt_auth(self):
        # Register a user
        uid = "test_uid"
        username = "test_username"
        password = "test_password"
        self.pwd_manager.register_password(uid, password, username=username)

        # Authenticate with username and password to get JWT
        response = self.client.get("/test", headers={"X-Username": username, "X-Password": password})
        self.assertEqual(response.status_code, 200)
        self.assertIn("X-JWT", response.headers)

        # Use the JWT token to authenticate
        jwt_token = response.headers["X-JWT"]
        response = self.client.get("/test", headers={"X-JWT": jwt_token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "success"})

    def test_invalid_jwt(self):
        response = self.client.get("/test", headers={"X-JWT": "invalid_token"})
        self.assertEqual(response.status_code, 401)

    def test_missing_auth_headers(self):
        response = self.client.get("/test")
        self.assertEqual(response.status_code, 400)

    def test_invalid_username_password(self):
        response = self.client.get("/test", headers={"X-Username": "invalid", "X-Password": "invalid"})
        self.assertEqual(response.status_code, 401)


if __name__ == '__main__':
    unittest.main()
