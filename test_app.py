from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
at.button[0].click().run()
assert not at.exception, at.exception
print([ (m.label, m.value) for m in at.metric])
