import re
import pandas as pd


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
    
    @staticmethod
    def load_templates(template_path : str) -> list: 
        """
        Load event templates from a file.
        """
        templates = []
        with open(template_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    templates.append(EventTemplate(line))
        return templates

    @staticmethod
    def matchfile(template_file, log_file):
        """
        Match events in a log file against templates and write the results to an output file.
        """
        templates = EventTemplate.load_templates(template_file)
        log_file = open(log_file)
        df = pd.DataFrame(columns=['log', 'event type'])
        for line in log_file:
            line = line.strip()
            if line:
                match = False
                for template in templates:
                    if template.match(line):
                        df = df._append({'log': line, 'event type': template.template}, ignore_index=True)
                        match = True
                        break
                if not match:
                    df = df._append({'log': line, 'event type': None}, ignore_index=True)
        log_file.close()
        return df

    @staticmethod
    def check_duplicate(template_file, log_file):
        """
        Check if a log file matches multiple templates.
        """
        templates = EventTemplate.load_templates(template_file)
        log_file = open(log_file)
        duplicate = False
        for line in log_file:
            line = line.strip()
            if line:
                matches = []
                for template in templates:
                    if template.match(line):
                        matches.append(template.template)
                if len(matches) > 1:
                    duplicate = True
                    print(f"[WARN] Duplicate found!")
                    print(f"log: `{line}`")
                    for match in matches:
                        print(f"template: `{match}`")
        log_file.close()

        return duplicate

    @staticmethod
    def completeness(template_file, log_file):
        """check if all logs are match"""
        log_file = open(log_file)
        templates = EventTemplate.load_templates(template_file)
        completeness = True

        for log in log_file:
            log = log.strip()
            if log:
                match = False
                for template in templates:
                    if template.match(log):
                        match = True
                        break
                if not match:
                    print(f"Not matched: `{log}`")
                    completeness = False

        log_file.close()
        return completeness

