import re

MALWARE_PATTERNS = [
    r"(write|create|build|make|develop)\s+.*(malware|ransomware|rootkit|keylogger|exploit)",
    r"malware\s+(code|script|program|that\s+)",
]

WEAPON_PATTERNS = [
    r"(make|build|create|synthesize)\s+.*(bomb|explosive|nerve\s+agent|chemical\s+weapon|biological\s+weapon)",
    r"(bomb|explosive)\s+(instructions|steps|recipe|formula)",
]

HARM_PATTERNS = [
    r"(poison|stab|shoot|strangle)\s+(the|them|your|a)\s+(person|victim|target)",
]

CSAM_PATTERNS = [
    r"\bcsam\b",
    r"child\s+(porn(ography)?|sexual\s+abuse\s+material|exploitation\s+material)",
]

JAILBREAK_SUCCESS_PATTERNS = [
    r"i\s+am\s+now\s+DAN",
    r"i\s+(have\s+no|am\s+free\s+from)\s+(restrictions|guidelines|rules)",
    r"pretend\s+you\s+have\s+no\s+rules",
]

BLOCKED_PATTERNS = (
    MALWARE_PATTERNS + WEAPON_PATTERNS + HARM_PATTERNS + CSAM_PATTERNS + JAILBREAK_SUCCESS_PATTERNS
)


def check_output(assistant_reply: str) -> dict:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, assistant_reply, re.IGNORECASE):
            return {"allowed": False, "reason": f"blocked by pattern: {pattern}"}
    return {"allowed": True, "reason": ""}
