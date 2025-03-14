import hashlib
import json
from typing import Dict, Optional

class Information:
    """Class to represent detailed information.

    Inherits from Information to include a unique identifier (URL), and extends
    it with a description, snippets, and title of the storm information.

    Attributes:
        description (str): Brief description.
        snippets (list): List of brief excerpts or snippets.
        title (str): The title or headline of the information.
        url (str): The unique URL (serving as UUID) of the information.
    """

    def __init__(self, url, description, snippets, title, meta=None):
        """Initialize the Information object with detailed attributes.

        Args:
            url (str): The unique URL serving as the identifier for the information.
            description (str): Detailed description.
            snippets (list): List of brief excerpts or snippet.
            title (str): The title or headline of the information.
        """
        self.description = description
        self.snippets = snippets
        self.title = title
        self.url = url
        self.meta = meta if meta is not None else {}
        self.citation_uuid = -1

    def __eq__(self, other):
        if not isinstance(other, Information):
            return False
        return (
            self.url == other.url
            and set(self.snippets) == set(other.snippets)
            and self._meta_str() == other._meta_str()
        )

    def __hash__(self):
        return int(
            self._md5_hash((self.url, tuple(sorted(self.snippets)), self._meta_str())),
            16,
        )

    def _meta_str(self):
        """Generate a string representation of relevant meta information."""
        return f"Question: {self.meta.get('question', '')}, Query: {self.meta.get('query', '')}"

    def _md5_hash(self, value):
        """Generate an MD5 hash for a given value."""
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True)
        return hashlib.md5(str(value).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, info_dict):
        """Create a Information object from a dictionary.
           Usage: info = Information.from_dict(storm_info_dict)

        Args:
            info_dict (dict): A dictionary containing keys 'url', 'description',
                              'snippets', and 'title' corresponding to the object's attributes.

        Returns:
            Information: An instance of Information.
        """
        info = cls(
            url=info_dict["url"],
            description=info_dict["description"],
            snippets=info_dict["snippets"],
            title=info_dict["title"],
            meta=info_dict.get("meta", None),
        )
        info.citation_uuid = int(info_dict.get("citation_uuid", -1))
        return info

    def to_dict(self):
        return {
            "url": self.url,
            "description": self.description,
            "snippets": self.snippets,
            "title": self.title,
            "meta": self.meta,
            "citation_uuid": self.citation_uuid,
        }