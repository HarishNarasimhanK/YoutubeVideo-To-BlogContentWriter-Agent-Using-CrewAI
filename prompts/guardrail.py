GUARDRAIL_SYSTEM = """\
You are an educational video classifier. Determine if the transcript is from an INFORMATIVE, educational, or explanatory video, or a NON-INFORMATIVE entertainment/vlog/sports video.
Respond ONLY with a single JSON object on one line:
{"verdict": "INFORMATIVE", "reason": "<brief reason>"}
or
{"verdict": "NON_INFORMATIVE", "reason": "<brief reason>"}
"""

GUARDRAIL_REJECTION_TEMPLATE = """\
Thank you for using the YouTube Video Demystifier! 🙏

Unfortunately, we were unable to process this video because it does not appear to be an educational or informative video.

**Classification reason**: {reason}

---

Our system is designed to work exclusively with educational content. Please try again with an educational or technical video! 🎓
"""
