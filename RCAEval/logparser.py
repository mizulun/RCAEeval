import re


class EventTemplate:
    """
    A class to represent an event template for matching events.
    """
    def __init__(self, template: str):
        self.event_id = None
        self.template = template
        self.regex = self._compile_template(template)

    def _compile_template(self, template: str) -> re.Pattern:
        """
        Compile the template into a regex pattern.
        """
        # Escape special characters and replace placeholders with regex patterns
        escaped_template = re.escape(template)
        # Replace <*> with a regex pattern that matches any word
        regex_pattern = escaped_template.replace("<\\*>", ".*?")
        # Compile the regex pattern
        return re.compile(regex_pattern)

    def match(self, event: str) -> bool:
        """
        Check if the event matches the template.
        """
        return bool(self.regex.match(event))




