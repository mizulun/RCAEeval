""" To debug regex: https://www.debuggex.com/ """
import json
import re
import toml
import pandas as pd
from collections import OrderedDict


def find_json_bounds(text):
    """Find (start,end) positions of all outermost {} pairs"""
    bounds = []
    start = None
    bracket_stack = 0

    for i, char in enumerate(text):
        if char == '{':
            if start is None:  # New opening
                start = i
            bracket_stack += 1
        elif char == '}' and bracket_stack > 0:
            bracket_stack -= 1 
            if bracket_stack == 0:
                # Found a complete JSON object
                bounds.append((start, i + 1))
                start = None

    return bounds  # List of (start, end) tuples

def mask_non_dict_values(data):
    if isinstance(data, dict):
        # Process dictionaries while preserving order
        masked_dict = OrderedDict()
        for key, value in data.items():
            masked_dict[key] = mask_non_dict_values(value)
        return masked_dict
    elif isinstance(data, (list, tuple)):
        # Process lists/tuples (mask non-dict elements, recurse into dict elements)
        return [mask_non_dict_values(item) for item in data]
    else:
        # Mask all non-dict values
        return "<*>"

def mask_values(data):
    if isinstance(data, dict):
        return OrderedDict((k, mask_values(v)) for k, v in data.items())
    elif isinstance(data, list):
        return [mask_values(item) for item in data]
    else:
        return "<*>"

#def remove_dict_values(line):
#    try:
#        # First try to parse the entire line as JSON
#        data = json.loads(line, object_pairs_hook=OrderedDict)
#        return json.dumps(mask_values(data))
#    except json.JSONDecodeError:
#        # If whole line isn't JSON, look for JSON substrings
#        parts = []
#        start = 0
#        
#        # Find all potential JSON segments in the line
#        for match in re.finditer(r'\{.*?\}', line):
#            # Add non-JSON text before the match
#            parts.append(line[start:match.start()])
#            
#            # Process the JSON segment
#            try:
#                data = json.loads(match.group(), object_pairs_hook=OrderedDict)
#                parts.append(json.dumps(mask_values(data)))
#            except json.JSONDecodeError:
#                parts.append(match.group())
#            
#            start = match.end()
#        
#        # Add remaining non-JSON text
#        parts.append(line[start:])
#        return ''.join(parts)

def remove_dict_values(text):
    """Process text and mask values in all JSON/dict objects"""
    bounds = find_json_bounds(text)
    if not bounds:
        return text
    
    for start, end in bounds:  
        json_str = text[start:end]
        import json
        data = json.loads(json_str.replace("'", "\""), object_pairs_hook=OrderedDict)
        masked = json.dumps(mask_values(data))
        
        # Replace original text segment
        text.replace(json_str, masked)
    
    return text

#def remove_dict_values(log_line):
#    # Pattern to find JSON-like dictionaries in the log line
#    pattern = r'(\{.*?\}(?=(?:(?:[^"]*"){2})*[^"]*$))'
#    
#    def replace_match(match):
#        try:
#            # Parse the JSON while preserving order
#            data = json.loads(match.group(1), object_pairs_hook=OrderedDict)
#            # Recursively mask all non-dict values
#            masked_data = mask_non_dict_values(data)
#            # Convert back to JSON string
#            return json.dumps(masked_data)
#        except json.JSONDecodeError:
#            # If it's not valid JSON, return the original match
#            return match.group(1)
#    
#    # Find and replace all JSON-like dictionaries in the line
#    return re.sub(pattern, replace_match, log_line)

#def remove_dict_values(log_line):
#    """Remove values from JSON-like dictionaries in the log line.
#    Example:
#    log_line: "Received ad request (context_words=[{'key': 'value'}])"
#    output = "Receved ad request (context_words=[{'key': '<*>'}])"
#    """
#    pattern = r'(\{.*?\})'
#
#    def replace_match(match):
#        try:
#            # Parse the JSON while preserving order
#            data = json.loads(match.group(1), object_pairs_hook=OrderedDict)
#            # Replace all values with <*>
#            for key in data:
#                if isinstance(data[key], (list, dict)):
#                    data[key] = "<*>"
#                else:
#                    data[key] = "<*>"
#            # Convert back to JSON string
#            return json.dumps(data)
#        except json.JSONDecodeError:
#            # If it's not valid JSON, return the original match
#            return match.group(1)
#
#    # Find and replace all JSON-like dictionaries in the line
#    return re.sub(pattern, replace_match, log_line)


class EventTemplate:
    """
    A class to represent an event template for matching events.
    """
    verbose = False
    def __init__(self, id: str = None, template: str = None, known_regex: dict = None):
        if not template:
            raise ValueError("Template cannot be None")
        self.id = id 
        self.template = template
        try:
            self.regex = self._compile_template(template, known_regex)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern in template: {template}. Error: {e}")

    def _compile_template(self, template: str, known_regex : dict = None) -> re.Pattern:
        """
        Compile the template into a regex pattern.
        """
        # Escape special characters and replace placeholders with regex patterns
        escaped_template = re.escape(template)

        if known_regex is not None:
            for name, pattern in known_regex.items():
                escaped_template = escaped_template.replace(f"<:{name}:>", pattern)

        # Replace <*> with a regex pattern that matches any word
        regex_pattern = escaped_template.replace("<\\*>", ".*?") + "$"
        if self.verbose:
            print(self.id, regex_pattern)
        # Compile the regex pattern
        return re.compile(regex_pattern)

    def is_match(self, event: str) -> bool:
        """
        Check if the event matches the template.
        """
        return bool(self.regex.match(event))
    
    @staticmethod
    def load_templates_from_txt(template_file: str) -> list: 
        """ Load event templates from a txt file.  """
        templates = []
        with open(template_file) as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    templates.append(EventTemplate(template=line))
        return templates

    @staticmethod
    def load_templates_from_toml(template_file):
        """Read templates from toml file"""
        templates = []

        with open(template_file) as f:
            config = toml.load(f)
        regex_patterns = config.get("Regex", {})

        event_types = config.get("EventTemplate", {}) 
        for event_id, template in event_types.items():
            template = EventTemplate(id=event_id, template=template, known_regex=regex_patterns)
            templates.append(template)

        return templates

    @staticmethod 
    def load_templates(template_file):
        if template_file.endswith(".toml"):
            return EventTemplate.load_templates_from_toml(template_file)
        elif template_file.endswith(".txt"):
            return EventTemplate.load_templates_from_txt(template_file)
        else:
            raise ValueError(f"Unsupported template file format: {template_file}. Supported formats are .toml and .txt")

    @staticmethod
    def parse_logs(template_file, log_file):
        """
        Parse logs and match them with templates.
        Args:
            template_file (str): Path to the template file.
            log_file (str): Path to the log file.
        """
        templates = EventTemplate.load_templates(template_file=template_file)
        log_file = open(log_file)
        df = pd.DataFrame(columns=['log', 'event type'])
        for line in log_file:
            line = line.strip()
            if line:
                match = False
                for template in templates:
                    if template.is_match(line):
                        df = df._append({'log': line, 'event type': template.template}, ignore_index=True)
                        match = True
                        break
                if not match:
                    df = df._append({'log': line, 'event type': None}, ignore_index=True)
        log_file.close()
        return df

    @staticmethod
    def is_duplicate(template_file, log_file):
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
                    if template.is_match(line):
                        matches.append(template.template)
                if len(matches) > 1:
                    duplicate = True
                    print(f"[WARN] Duplicate found!")
                    print(f"log: `{line}`")
                    for match in matches:
                        print(f"template: `{match}`")
        log_file.close()
        if not duplicate:
            print(f"[INFO] No duplicate found.")
        return duplicate
    
    @staticmethod
    def is_complete(template_file, log_file):
        """check if all logs are match"""
        log_file = open(log_file)
        templates = EventTemplate.load_templates(template_file)
        completeness = True
        match_count = 0
        not_match_count = 0
        not_match_logs = []

        for log in log_file:
            log = log.strip()
            if log:
                match = False
                for template in templates:
                    if template.is_match(log):
                        match = True
                        break
                if not match:
                    not_match_logs.append(log)
                    not_match_count += 1
                    completeness = False
                else: 
                    match_count += 1

        log_file.close()

        if completeness:
            print(f"[INFO] All logs are matched. {match_count} logs matched.")
        else:
            print(f"[INFO] {match_count/(match_count + not_match_count)*100:.2f}% logs matched.")
            print(f"[WARN] {not_match_count} logs not matched.")
            print(f"[INFO] Not matched logs:")
            for log in not_match_logs:
                print(log)
        return completeness



#output = mask_dict_values("This is a log: {'id': 100, 'data': {'log-1': 'aaa', 10: {'this is a set'}}} with some values.")
#print(output)
