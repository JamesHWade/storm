import dspy
import numpy as np
import re
import threading
from typing import Set, Dict, List, Optional, Union, Tuple

from .encoder import Encoder
from .interface import Information


class ConversationTurn:
    """
    A class to represent a turn in a conversation.

    Attributes:
        role (str): A short phrase of the role of the speaker for the current conversation turn.
        raw_utterance (str): The response generated by the LM model without polished style and tone.
        utterance_type (str): The type of utterance (e.g., statement, question).
        claim_to_make (Optional[str]): The point that this utterance tries to make. Should be empty if the utterance type is questioning.
        utterance (Optional[str]): The response generated by the model with polished style and tone. Defaults to raw_utterance if not provided.
        queries (List[str]): The queries used to gather information to have a grounded answer.
        raw_retrieved_info (List['Information']): A list of Information type that is retrieved.
        cited_info (Dict[int, 'Information']): A dictionary where the key is the citation index and the value is Information type.
        role_description (Optional[str]): A few sentences description of the role. Defaults to an empty string if not provided.
    """

    def __init__(
        self,
        role: str,
        raw_utterance: str,
        utterance_type: str,
        claim_to_make: Optional[str] = None,
        utterance: Optional[str] = None,
        queries: Optional[List[str]] = None,
        raw_retrieved_info: Optional[List[Information]] = None,
        cited_info: Optional[List[Information]] = None,
    ):
        self.utterance = utterance if utterance is not None else raw_utterance
        self.raw_utterance = raw_utterance
        self.role = role if ":" not in role else role.split(":")[0]
        self.role_description = "" if ":" not in role else role.split(":")[1]
        self.queries = queries if queries is not None else []
        self.raw_retrieved_info = (
            raw_retrieved_info if raw_retrieved_info is not None else []
        )
        self.cited_info = cited_info if cited_info is not None else {}
        self.utterance_type = utterance_type
        self.claim_to_make = claim_to_make if claim_to_make is not None else ""

    def get_all_citation_index(self):
        citation_pattern = re.compile(r"\[(\d+)\]")
        return list(map(int, citation_pattern.findall(self.utterance)))

    def to_dict(self):
        raw_retrieved_info = [info.to_dict() for info in self.raw_retrieved_info]
        return {
            "utterance": self.utterance,
            "raw_utterance": self.raw_utterance,
            "role": self.role,
            "role_description": self.role_description,
            "queries": self.queries,
            "utterance_type": self.utterance_type,
            "claim_to_make": self.claim_to_make,
            "raw_retrieved_info": raw_retrieved_info,
            "cited_info": None,
        }

    @classmethod
    def from_dict(cls, conv_turn_dict: Dict):
        raw_retrieved_info = [
            Information.from_dict(info) for info in conv_turn_dict["raw_retrieved_info"]
        ]

        return cls(
            utterance=conv_turn_dict["utterance"],
            raw_utterance=conv_turn_dict["raw_utterance"],
            role=f"{conv_turn_dict['role']}: {conv_turn_dict['role_description']}",
            queries=conv_turn_dict["queries"],
            raw_retrieved_info=raw_retrieved_info,
            cited_info=None,
            utterance_type=conv_turn_dict["utterance_type"],
            claim_to_make=conv_turn_dict["claim_to_make"],
        )


class KnowledgeNode:
    """
    Class representing a node in the knowledge base.

    Attributes:
        name (str): The name of the node.
        content (list): A list of Information instances.
        children (list): A list of child KnowledgeNode instances.
        parent (KnowledgeNode): The parent node of the current node.
    """

    def __init__(
        self,
        name: str,
        content: Optional[str] = None,
        parent: Optional["KnowledgeNode"] = None,
        children: Optional[List["KnowledgeNode"]] = None,
        synthesize_output: Optional[str] = None,
        need_regenerate_synthesize_output: bool = True,
    ):
        """
        Initializes a KnowledgeNode instance.

        Args:
            name (str): The name of the node.
            content (list, optional): A list of information uuid. Defaults to None.
            parent (KnowledgeNode, optional): The parent node of the current node. Defaults to None.
        """
        self.name = name
        self.content: Set[int] = set(content) if content is not None else set()
        self.children = [] if children is None else children
        self.parent = parent
        self.synthesize_output = synthesize_output
        self.need_regenerate_synthesize_output = need_regenerate_synthesize_output

    def collect_all_content(self):
        """
        Collects all content from the current node and its descendants.

        Returns:
            Set[int]: A set containing all content from the current node and its descendants.
        """
        all_content = set(self.content)
        for child in self.children:
            all_content.update(child.collect_all_content())
        return all_content

    def has_child(self, child_node_name: str):
        """
        Check if the node has the child of given name.
        """
        return child_node_name in [child.name for child in self.children]

    def add_child(self, child_node_name: str, duplicate_handling: str = "skip"):
        """
        Adds a child node to the current node.
        duplicate_handling (str): How to handle duplicate nodes. Options are "skip", "none", and "raise error".
        """
        if self.has_child(child_node_name):
            if duplicate_handling == "skip":
                for child in self.children:
                    if child.name == child_node_name:
                        return child
            elif duplicate_handling == "raise error":
                raise Exception(
                    f"Insert node error. Node {child_node_name} already exists under its parent node {self.name}."
                )
        child_node = KnowledgeNode(name=child_node_name, parent=self)
        self.children.append(child_node)
        return child_node

    def get_parent(self):
        """
        Returns the parent node of the current node.

        Returns:
            KnowledgeNode: The parent node of the current node.
        """
        return self.parent

    def get_children(self):
        """
        Returns the children of the current node.

        Returns:
            list: A list of child KnowledgeNode instances.
        """
        return self.children

    def get_children_names(self):
        """
        Returns a list of children names.
        """
        return [child.name for child in self.children]

    def __repr__(self):
        """
        Returns a string representation of the KnowledgeNode instance.

        Returns:
            str: String representation of the KnowledgeNode instance.
        """
        return f"KnowledgeNode(name={self.name}, content={self.content}, children={len(self.children)})"

    def get_path_from_root(self, root: Optional["KnowledgeNode"] = None):
        """
        Get a list of names from the root to this node.

        Returns:
            List[str]: A list of node names from the root to this node.
        """
        path = []
        current_node = self
        while current_node:
            path.append(current_node.name)
            if root is not None and current_node.name == root.name:
                break
            current_node = current_node.parent
        return path[::-1]

    def insert_information(self, information_index: int):
        if information_index not in self.content:
            self.need_regenerate_synthesize_output = True
            self.content.add(information_index)

    def get_all_descendents(self) -> List["KnowledgeNode"]:
        """
        Get a list of all descendant nodes.

        Returns:
            List[KnowledgeNode]: A list of all descendant nodes.
        """
        descendents = []

        def collect_descendents(node):
            for child in node.children:
                descendents.append(child)
                collect_descendents(child)

        collect_descendents(self)
        return descendents

    def get_all_predecessors(self) -> List["KnowledgeNode"]:
        """
        Get a list of all predecessor nodes (from current node to root).

        Returns:
            List[KnowledgeNode]: A list of all predecessor nodes.
        """
        predecessors = []
        current_node = self.parent
        while current_node is not None:
            predecessors.append(current_node)
            current_node = current_node.parent
        return predecessors

    def to_dict(self):
        """
        Converts the KnowledgeNode instance to a dictionary representation.

        Returns:
            dict: The dictionary representation of the KnowledgeNode.
        """
        return {
            "name": self.name,
            "content": list(self.content),
            "children": [child.to_dict() for child in self.children],
            "parent": self.parent.name if self.parent else None,
            "synthesize_output": self.synthesize_output,
            "need_regenerate_synthesize_output": self.need_regenerate_synthesize_output,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Constructs a KnowledgeNode instance from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the KnowledgeNode.

        Returns:
            KnowledgeNode: The constructed KnowledgeNode instance.
        """

        def helper(cls, data, parent_node=None):
            if parent_node is not None:
                assert data["parent"] is not None and data["parent"] == parent_node.name
            node = cls(
                name=data["name"],
                content=data["content"],
                parent=parent_node,
                children=None,
                synthesize_output=data.get("synthesize_output", None),
                need_regenerate_synthesize_output=data.get(
                    "need_regenerate_synthesize_output", True
                ),
            )
            for child_data in data["children"]:
                child_node = helper(cls, child_data, parent_node=node)
                node.children.append(child_node)
            return node

        return helper(cls, data)


class KnowledgeBase:
    """
    Represents the dynamic, hierarchical mind map used in Co-STORM to track and organize discourse.

    The knowledge base serves as a shared conceptual space between the user and the system, allowing for effective collaboration by reducing the user's cognitive load and ensuring that the discourse is easy to follow.

    The knowledge base is structured as a tree (or mind map) that dynamically organizes collected information and concepts as the conversation progresses.

    The mind map consists of concepts (nodes) and edges that represent parent-child relationships among topics. Each concept is linked to retrieved information,
    which is placed under the most appropriate concept based on its associated question and semantic similarity.

    For more details, please refer to Section 3.2 of Co-STORM paper: https://www.arxiv.org/pdf/2408.15232
    Attributes:
        root (KnowledgeNode): The root node of the hierarchical knowledge base, representing the top-level concept.

    """

    def __init__(
        self,
        topic: str,
        knowledge_base_lm: dspy.LM,
        node_expansion_trigger_count: int,
        encoder: Encoder,
    ):
        """
        Initializes a KnowledgeBase instance.

        Args:
            topic (str): The topic of the knowledge base
            expand_node_module (dspy.Module): The module that organize knowledge base in place.
                The module should accept knowledge base as param. E.g. expand_node_module(self)
            article_generation_module (dspy.Module): The module that generate report from knowledge base.
                The module should return string. E.g. report = article_generation_module(self)
        """
        from .collaborative_storm.modules.article_generation import (
            ArticleGenerationModule,
        )
        from .collaborative_storm.modules.information_insertion_module import (
            InsertInformationModule,
            ExpandNodeModule,
        )
        from .collaborative_storm.modules.knowledge_base_summary import (
            KnowledgeBaseSummaryModule,
        )

        self.topic: str = topic
        self.encoder: Encoder = encoder

        self.information_insert_module = InsertInformationModule(
            engine=knowledge_base_lm, encoder=self.encoder
        )
        self.expand_node_module = ExpandNodeModule(
            engine=knowledge_base_lm,
            information_insert_module=self.information_insert_module,
            node_expansion_trigger_count=node_expansion_trigger_count,
        )
        self.article_generation_module = ArticleGenerationModule(
            engine=knowledge_base_lm
        )
        self.gen_summary_module = KnowledgeBaseSummaryModule(engine=knowledge_base_lm)

        self.root: KnowledgeNode = KnowledgeNode(name="root")
        self.kb_embedding = {
            "hash": hash(""),
            "encoded_structure": np.array([[]]),
            "structure_string": "",
        }
        self.info_uuid_to_info_dict: Dict[int, Information] = {}
        self.info_hash_to_uuid_dict: Dict[int, int] = {}
        self._lock = threading.Lock()

    def to_dict(self):
        info_uuid_to_info_dict = {
            key: value.to_dict() for key, value in self.info_uuid_to_info_dict.items()
        }
        return {
            "topic": self.topic,
            "tree": self.root.to_dict(),
            "info_uuid_to_info_dict": info_uuid_to_info_dict,
            "info_hash_to_uuid_dict": self.info_hash_to_uuid_dict,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict,
        knowledge_base_lm: dspy.LM,
        node_expansion_trigger_count: int,
        encoder: Encoder,
    ):
        knowledge_base = cls(
            topic=data["topic"],
            knowledge_base_lm=knowledge_base_lm,
            node_expansion_trigger_count=node_expansion_trigger_count,
            encoder=encoder,
        )
        knowledge_base.root = KnowledgeNode.from_dict(data["tree"])
        knowledge_base.info_hash_to_uuid_dict = {
            int(key): int(value)
            for key, value in data["info_hash_to_uuid_dict"].items()
        }
        info_uuid_to_info_dict = {
            int(key): Information.from_dict(value)
            for key, value in data["info_uuid_to_info_dict"].items()
        }
        knowledge_base.info_uuid_to_info_dict = info_uuid_to_info_dict
        return knowledge_base

    def get_knowledge_base_structure_embedding(
        self, root: Optional[KnowledgeNode] = None
    ) -> Tuple[np.ndarray, List[str]]:
        outline_string = self.get_node_hierarchy_string(
            include_indent=False,
            include_full_path=True,
            include_hash_tag=False,
            root=root,
        )
        outline_string_hash = hash(outline_string)
        if outline_string_hash != self.kb_embedding["hash"]:
            outline_strings: List[str] = outline_string.split("\n")
            cleaned_outline_strings = [
                outline.replace(" -> ", ", ") for outline in outline_strings
            ]
            encoded_outline = self.encoder.encode(cleaned_outline_strings)
            self.kb_embedding = {
                "hash": outline_string_hash,
                "encoded_structure": encoded_outline,
                "structure_string": outline_strings,
            }
        return (
            self.kb_embedding["encoded_structure"],
            self.kb_embedding["structure_string"],
        )

    def traverse_down(self, node):
        """
        Traverses the tree downward from the given node.

        Args:
            node (KnowledgeNode): The node to start the traversal from.

        Returns:
            list: A list of KnowledgeNode instances in the order they were visited.
        """
        nodes = []

        def _traverse(current_node):
            nodes.append(current_node)
            for child in current_node.get_children():
                _traverse(child)

        _traverse(node)
        return nodes

    def traverse_up(self, node):
        """
        Traverses the tree upward from the given node.

        Args:
            node (KnowledgeNode): The node to start the traversal from.

        Returns:
            list: A list of KnowledgeNode instances in the order they were visited.
        """
        nodes = []
        while node is not None:
            nodes.append(node)
            node = node.get_parent()
        return nodes

    def collect_all_nodes(self):
        nodes = []

        def _collect(node):
            nodes.append(node)
            for child in node.children:
                _collect(child)

        _collect(self.root)
        return nodes

    def insert_node(
        self,
        new_node_name,
        parent_node: Optional[KnowledgeNode] = None,
        duplicate_handling="skip",
    ):
        """
        Inserts a new node into the knowledge base under the specified parent node.

        Args:
            new_node_name (str): The name of the new node.
            parent_node_name (str): The name of the parent node. If None, the new node is inserted under the root.
            duplicate_handling (str): How to handle duplicate nodes. Options are "skip", "none", and "raise error".
        """
        if parent_node is None:
            return self.root.add_child(
                new_node_name, duplicate_handling=duplicate_handling
            )
        else:
            return parent_node.add_child(
                new_node_name, duplicate_handling=duplicate_handling
            )

    def find_node(self, current_node, node_name):
        """
        Finds a node by name in the knowledge base.

        Args:
            current_node (KnowledgeNode): The node to start the search from.
            node_name (str): The name of the node to find.

        Returns:
            KnowledgeNode: The node with the specified name, or None if not found.
        """
        if current_node.name == node_name:
            return current_node
        for child in current_node.get_children():
            result = self.find_node(child, node_name)
            if result is not None:
                return result
        return None

    def insert_from_outline_string(self, outline_string, duplicate_handling="skip"):
        """
        Creates and inserts nodes into the knowledge base from a string outline.

        Args:
            outline_string (str): The outline string where each line starts with '#' denoting the level.
            duplicate_handling (str): How to handle duplicate nodes. Options are "skip", "none", and "raise error".
        """
        last_node_at_level = {}
        for line in outline_string.split("\n"):
            level = line.count("#")
            if level > 0:
                title = line.strip("# ").strip()
                if title.lower() in ["overview", "summary", "introduction"]:
                    continue
                parent_node = None if level == 1 else last_node_at_level.get(level - 1)
                new_node = self.insert_node(
                    new_node_name=title,
                    parent_node=parent_node,
                    duplicate_handling=duplicate_handling,
                )
                last_node_at_level[level] = new_node
                for deeper_level in list(last_node_at_level.keys()):
                    if deeper_level > level:
                        del last_node_at_level[deeper_level]

    def get_node_hierarchy_string(
        self,
        include_indent=False,
        include_full_path=False,
        include_hash_tag=True,
        include_node_content_count=False,
        cited_indices: Optional[List[int]] = None,
        root: Optional[KnowledgeNode] = None,
    ) -> str:
        def find_node_contain_index(node, index):
            """
            Traverses the tree downward from the given node.

            Args:
                node (KnowledgeNode): The node to start the traversal from.

            Returns:
                list: A list of KnowledgeNode instances in the order they were visited.
            """
            nodes = []

            def _traverse(current_node):
                if current_node is not None and index in current_node.content:
                    nodes.append(current_node)
                for child in current_node.get_children():
                    _traverse(child)

            _traverse(node)
            return nodes

        paths_to_highlight = set()
        nodes_to_include = set()
        if cited_indices is not None:
            for index in cited_indices:
                for cur_node in find_node_contain_index(self.root, index):
                    paths_to_highlight.add(" -> ".join(cur_node.get_path_from_root()))
                    nodes_to_include.add(cur_node)
                    nodes_to_include.update(cur_node.get_all_descendents())
                    predecessors = cur_node.get_all_predecessors()
                    for predecessor in predecessors:
                        nodes_to_include.update(predecessor.children)
                    nodes_to_include.update(predecessors)

        def should_include_node(node):
            if cited_indices is None:
                return True
            return node in nodes_to_include

        def should_omit_child_nodes(node):
            if cited_indices is None:
                return False
            for child in node.children:
                if should_include_node(child):
                    return False
            return True

        def helper(cur_root, level):
            to_return = []
            if cur_root is not None:
                should_include_current_node = should_include_node(cur_root)

                indent = "" if not include_indent else "\t" * (level - 1)
                full_path = " -> ".join(cur_root.get_path_from_root(root=root))
                node_info = cur_root.name if not include_full_path else full_path
                hash_tag = "#" * level + " " if include_hash_tag else ""
                content_count = (
                    f" ({len(cur_root.content)})" if include_node_content_count else ""
                )
                special_note = (
                    ""
                    if cited_indices is None or full_path not in paths_to_highlight
                    else " ⭐"
                )

                if should_include_current_node:
                    to_return.append(
                        f"{indent}{hash_tag}{node_info}{content_count}{special_note}"
                    )
                    if should_omit_child_nodes(cur_root):
                        if len(cur_root.children) > 0:
                            child_indent = indent = (
                                "" if not include_indent else "\t" * (level)
                            )
                            to_return.append(f"{child_indent}...")
                    else:
                        for child in cur_root.children:
                            to_return.extend(helper(child, level + 1))
            return to_return

        to_return = []
        if root is None and self.root is not None:
            for child in self.root.children:
                to_return.extend(helper(child, level=1))
        else:
            to_return.extend(helper(root, level=1))

        return "\n".join(to_return)

    def find_node_by_path(
        self,
        path: str,
        missing_node_handling="abort",
        root: Optional[KnowledgeNode] = None,
    ):
        """
        Returns the target node given a path string.

        Args:
            path (str): The path to the node, with node names connected by " -> ".
            missing_node_handling (str): How to handle missing nodes. Options are "abort", "create", and "raise error".

        Returns:
            KnowledgeNode: The target node.
        """
        node_names = path.split(" -> ")
        current_node = self.root if root is None else root

        for name in node_names[1:]:
            found_node = next(
                (child for child in current_node.children if child.name == name), None
            )
            if found_node is None:
                if missing_node_handling == "abort":
                    return
                elif missing_node_handling == "create":
                    new_node = current_node.add_child(child_node_name=name)
                    current_node = new_node
                elif missing_node_handling == "raise error":
                    structure = self.get_node_hierarchy_string(
                        include_indent=True,
                        include_full_path=False,
                        include_hash_tag=True,
                    )
                    raise Exception(
                        f"Insert information error. Unable to find node {{{name}}} under {{{current_node.name}}}\n{structure}"
                    )
            else:
                current_node = found_node
        return current_node

    def insert_information(
        self,
        path: str,
        information: Information,
        missing_node_handling="abort",
        root: Optional[KnowledgeNode] = None,
    ):
        """
        Inserts information into the knowledge base at the specified path.

        Args:
            path (str): The placement path string, connected by " -> " linking the name of nodes.
            information (Information): The information to insert.
            missing_node_handling (str): How to handle missing nodes. Options are "abort", "create", and "raise error".
        Return:
            uuid of insertion information
        """
        with self._lock:
            target_node: KnowledgeNode = self.find_node_by_path(
                path=path, missing_node_handling=missing_node_handling, root=root
            )
            information_hash = hash(information)
            if information.citation_uuid == -1:
                info_citation_uuid = self.info_hash_to_uuid_dict.get(
                    information_hash, len(self.info_hash_to_uuid_dict) + 1
                )
                information.citation_uuid = info_citation_uuid
                self.info_hash_to_uuid_dict[information_hash] = info_citation_uuid
                self.info_uuid_to_info_dict[info_citation_uuid] = information
            if target_node is not None:
                self.info_uuid_to_info_dict[information.citation_uuid].meta[
                    "placement"
                ] = " -> ".join(target_node.get_path_from_root())
                target_node.insert_information(information.citation_uuid)

    def trim_empty_leaf_nodes(self):
        """
        Trims all leaf nodes that do not have any content. Iteratively does it until all leaf nodes have at least one content.
        """

        def trim_node(node):
            if not node.children and not node.content:
                return True
            node.children = [child for child in node.children if not trim_node(child)]
            return not node.children and not node.content

        # Start the trimming process from the root
        while True:
            before_trim = len(self.get_all_leaf_nodes())
            trim_node(self.root)
            after_trim = len(self.get_all_leaf_nodes())
            if before_trim == after_trim:
                break

    def get_all_leaf_nodes(self):
        """
        Helper function to get all leaf nodes.

        Returns:
            List[KnowledgeNode]: A list of all leaf nodes in the knowledge base.
        """
        leaf_nodes = []

        def find_leaf_nodes(node):
            if not node.children:
                leaf_nodes.append(node)
            for child in node.children:
                find_leaf_nodes(child)

        find_leaf_nodes(self.root)
        return leaf_nodes

    def merge_single_child_nodes(self):
        """
        Merges content of a node with its single child and removes the child node.
        Iteratively does this from leaf nodes back to the root.
        """

        def merge_node(node):
            # Recursively merge children first
            for child in node.children:
                merge_node(child)

            # If the node has exactly one child, merge its content with the child and remove the child
            if len(node.children) == 1:
                single_child = node.children[0]
                node.content.update(single_child.content)
                node.children = single_child.children
                for grandchild in node.children:
                    grandchild.parent = node

        merge_node(self.root)

    def update_all_info_path(self):
        def _helper(node):
            for citation_idx in node.content:
                self.info_uuid_to_info_dict[citation_idx].meta["placement"] = (
                    " -> ".join(node.get_path_from_root())
                )
            for child in node.children:
                _helper(child)

        _helper(self.root)

    def update_from_conv_turn(
        self,
        conv_turn: ConversationTurn,
        allow_create_new_node: bool = False,
        insert_under_root: bool = False,
    ):
        if conv_turn is None:
            return
        info_to_insert = list(conv_turn.cited_info.values())
        if insert_under_root:
            for info in info_to_insert:
                self.insert_information(path=self.root.name, information=info)
        else:
            self.information_insert_module(
                knowledge_base=self,
                information=info_to_insert,
                allow_create_new_node=allow_create_new_node,
            )
        old_to_new_citation_idx_mapping = {
            old_idx: info.citation_uuid
            for old_idx, info in conv_turn.cited_info.items()
        }

        for old_idx, new_idx in old_to_new_citation_idx_mapping.items():
            conv_turn.utterance = conv_turn.utterance.replace(
                f"[{old_idx}]", f"[_{new_idx}_]"
            )
            conv_turn.raw_utterance = conv_turn.raw_utterance.replace(
                f"[{old_idx}]", f"[_{new_idx}_]"
            )
        for _, new_idx in old_to_new_citation_idx_mapping.items():
            conv_turn.utterance = conv_turn.utterance.replace(
                f"[_{new_idx}_]", f"[{new_idx}]"
            )
            conv_turn.utterance.replace("[-1]", "")
            conv_turn.raw_utterance = conv_turn.raw_utterance.replace(
                f"[_{new_idx}_]", f"[{new_idx}]"
            )
            conv_turn.raw_utterance.replace("[-1]", "")
        conv_turn.cited_info = None

    def get_knowledge_base_summary(self):
        return self.gen_summary_module(self)

    def reorganize(self):
        """
        Reorganizes the knowledge base through two main processes: top-down expansion and bottom-up cleaning.

        The reorganization process ensures that the knowledge base remains well-structured and relevant as new information is added. It consists of the following steps:
        1.Top-Down Expansion: Expands nodes that have accumulated significant amounts of information by creating subtopics,
          ensuring that each concept remains specific and manageable.
        2.Bottom-Up Cleaning: Cleans the knowledge base by removing empty leaf nodes (nodes with no supporting information)
          and merging nodes that have only a single child, simplifying the structure and maintaining clarity.
        """
        # pre-processing
        self.trim_empty_leaf_nodes()
        self.merge_single_child_nodes()
        # expand nodes
        self.expand_node_module(knowledge_base=self)
        # clean up
        self.trim_empty_leaf_nodes()
        self.merge_single_child_nodes()
        self.update_all_info_path()

    def to_report(self):
        return self.article_generation_module(knowledge_base=self)
