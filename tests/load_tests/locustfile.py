"""Locust load tests for URCM API."""
from locust import HttpUser, task, between
import json


class URCMHealthUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def metrics(self):
        self.client.get("/metrics")


class URCMReasonUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.headers = {"Content-Type": "application/json"}

    @task(3)
    def short_reason(self):
        self.client.post(
            "/api/reason",
            json={"text": "What is truth?", "max_steps": 10},
            headers=self.headers,
        )

    @task(2)
    def medium_reason(self):
        self.client.post(
            "/api/reason",
            json={"text": "Explain the relationship between consciousness and reality.", "max_steps": 20},
            headers=self.headers,
        )

    @task(1)
    def long_reason(self):
        self.client.post(
            "/api/reason",
            json={"text": "Describe the fundamental nature of mathematics and its relationship to physical reality. Consider Platonism, formalism, and intuitionism as philosophical frameworks.", "max_steps": 30},
            headers=self.headers,
        )

    @task(1)
    def validate(self):
        self.client.get("/api/validate")
