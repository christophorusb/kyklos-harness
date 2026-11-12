"""Background jobs."""

RETRY_LIMIT = 5
JOBS = []


def enqueue(name, payload):
    JOBS.append({"name": name, "payload": payload, "tries": 0})


def give_up(job):
    return job["tries"] >= RETRY_LIMIT
