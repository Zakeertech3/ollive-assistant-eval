import re

MALWARE_PATTERNS = [
    r"(write|create|build|make|develop|code)\s+.*(malware|ransomware|rootkit|keylogger|spyware|exploit\s+code)",
]

WEAPON_PATTERNS = [
    r"(how\s+to|instructions?\s+(for|to)|steps?\s+(to|for))\s+.*(make|build|create|synthesize)\s+.*(bomb|explosive|nerve\s+agent|sarin|chemical\s+weapon|biological\s+weapon|anthrax|ricin)",
    r"(make|build|create)\s+.*(pipe\s+bomb|car\s+bomb|dirty\s+bomb)",
]

HARM_PATTERNS = [
    r"how\s+to\s+(kill|murder|harm|hurt|poison|assault)\s+(a\s+specific\s+|someone\s+named\s+|my\s+)?(person|people|someone|individual|neighbor|coworker|family\s+member)",
]

CSAM_PATTERNS = [
    r"\bcsam\b",
    r"child\s+(porn(ography)?|sexual\s+abuse\s+material|exploitation\s+material)",
    r"(sexual|explicit)\s+content\s+(of|involving|with)\s+(a\s+)?(minor|child|underage)",
]

JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+instructions",
    r"you\s+are\s+now\s+DAN",
    r"pretend\s+you\s+have\s+no\s+rules",
    r"act\s+as\s+if\s+you\s+have\s+no\s+(restrictions|rules|guidelines|limitations)",
    r"(disregard|forget|bypass)\s+(all\s+)?(previous|prior|your)?\s*(instructions|guidelines|rules|training|constraints)",
]

BLOCKED_PATTERNS = (
    MALWARE_PATTERNS + WEAPON_PATTERNS + HARM_PATTERNS + CSAM_PATTERNS + JAILBREAK_PATTERNS
)


def check_input(user_message: str) -> dict:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, user_message, re.IGNORECASE):
            return {"allowed": False, "reason": f"blocked by pattern: {pattern}"}
    return {"allowed": True, "reason": ""}
