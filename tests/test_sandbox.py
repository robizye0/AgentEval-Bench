import asyncio

from agenteval.mock.sandbox import sandbox_state


def test_sandbox_context_rolls_back_mapping_on_success_and_error():
    state = {"rows": [1], "revision": 3}

    with sandbox_state(state):
        state["rows"].append(2)
        state["revision"] += 1

    assert state == {"rows": [1], "revision": 3}

    try:
        with sandbox_state(state):
            state["rows"].clear()
            raise RuntimeError("simulated tool error")
    except RuntimeError:
        pass
    assert state == {"rows": [1], "revision": 3}


def test_sandbox_decorator_rolls_back_sync_and_async_mock_mutations():
    state = {"records": []}

    @sandbox_state(state)
    def create_record(name):
        state["records"].append(name)
        return {"id": len(state["records"]), "name": name}

    @sandbox_state(state)
    async def async_create_record(name):
        state["records"].append(name)
        await asyncio.sleep(0)
        return name

    assert create_record("temporary") == {"id": 1, "name": "temporary"}
    assert asyncio.run(async_create_record("async")) == "async"
    assert state == {"records": []}


def test_sandbox_uses_explicit_snapshot_restore_protocol():
    class FakeDatabase:
        def __init__(self):
            self.rows = ["baseline"]

        def sandbox_snapshot(self):
            return list(self.rows)

        def sandbox_restore(self, snapshot):
            self.rows[:] = snapshot

    db = FakeDatabase()
    with sandbox_state(db):
        db.rows.append("uncommitted")
    assert db.rows == ["baseline"]
