import page
csrf = "89e140d03745ed5d840b7e85cd85fd7666f1868771b6f26685c89161757ad2f2"
session_data = page.evaluate(f"""async () => {{
  const res = await fetch(`/api/auth/session?csrfToken={csrf}`);
  return res.json();
}}""")
auth_code = session_data.auth0.authorization_code

print(f"authCode:{auth_code}")

token_data = page.evaluate(f"""async () => {{
  const form = new URLSearchParams();
  form.append("grant_type", "authorization_code");
  form.append("client_id", "codex专属client_id");
  form.append("code", "{auth_code}");
  form.append("redirect_uri", "https://chatgpt.com/api/auth/callback");
  const res = await fetch("https://auth0.openai.com/oauth/token", {{
    method: "POST",
    headers: {{"Content-Type":"application/x-www-form-urlencoded"}},
    body: form
  }});
  return res.json();
}}""")

print(f"token_data:{token_data}")