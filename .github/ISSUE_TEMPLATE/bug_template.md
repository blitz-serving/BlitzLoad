---
name: Bug report template
about: Report a reproducible bug
title: "[bug]"
labels: 'bug'
assignees: ''

---

## 🐞 Bug Description
<!-- A clear and concise description of the problem -->
Example: Login API returns 500 error when using valid credentials

## 🔄 Steps to Reproduce
<!-- Step-by-step instructions to reproduce the bug -->
1. Start backend service
2. Send POST request to `/api/auth/login` with valid credentials
3. Observe server response

## ✅ Expected Behavior
<!-- What you expected to happen -->
Example: Server should return `200 OK` with JWT token

## 🚨 Actual Behavior
<!-- What actually happened -->
Example: Server responds with `500 Internal Server Error`

## 🧪 Test Evidence / Logs
<!-- Screenshots, console logs, error traces, request payloads -->
POST /api/auth/login
Response: 500 Internal Server Error
Error: NullPointerException at AuthService.java:42


## 📌 Milestone / To-Do
- [ ] Debug `AuthService.login()` logic
- [ ] Check DB query for null values
- [ ] Add error handling for missing user
- [ ] Write regression test for login API

## 🧪 Required Unit Tests
```gherkin
Scenario: Login with valid user
  Given A user exists in the database
  When POST request with correct password
  Then Return 200 status and JWT token

Scenario: Login with null user
  Given Email not found in DB
  When POST request
  Then Return 404 status with error message

```
## 💻 Related Code
- `/src/services/AuthService.java`
- `/src/routes/auth.js`

## 💬 Notes and Comments
1. Might affect registration flow as well
2. Coordinate with DB team to verify schema consistency

## 📅 Schedule
Target fix: YYYY-MM-DD
