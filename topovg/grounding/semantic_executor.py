"""Original graph-to-attention semantic heuristics, unchanged."""
from ..utils.backend import LazyModule
from ..utils.text import normalize_text

torch = LazyModule('torch')

class SemanticExecutor:
    def __init__(self, tokenizer, cross_attn_map):
        self.tokenizer = tokenizer
        self.res = cross_attn_map.shape[0]
        self.cross_attn = cross_attn_map.permute(2, 0, 1)
        self.prompt_tokens = []

    def set_prompt(self, prompt):
        prompt = normalize_text(prompt)
        self.prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)[:77]

    def get_token_attn(self, word):
        clean_word = normalize_text(word).strip(" ,.!?;:'\"")
        word_tokens = self.tokenizer.encode(clean_word, add_special_tokens=False)

        attn_sum = torch.zeros((self.res, self.res)).to(self.cross_attn.device)
        valid_count = 0

        for wt in word_tokens:
            indices = [i for i, x in enumerate(self.prompt_tokens) if x == wt]
            for idx in indices:
                attn_sum += self.cross_attn[idx].float()
                valid_count += 1

        if valid_count > 0:
            return attn_sum / valid_count

        return torch.ones((self.res, self.res)).to(self.cross_attn.device)

    def execute_compound(self, words_list):
        if not words_list:
            return torch.ones((self.res, self.res)).to(self.cross_attn.device)

        fused_attn = torch.zeros((self.res, self.res)).to(self.cross_attn.device)
        for word in words_list:
            fused_attn += self.get_token_attn(word)
        return fused_attn

    def execute_attributes(self, base_attn, attributes_list):
        if not attributes_list:
            return base_attn

        attr_fused = torch.zeros((self.res, self.res)).to(self.cross_attn.device)
        for attr in attributes_list:
            attr_fused += self.get_token_attn(attr)

        return base_attn * attr_fused

    def generate_spatial_matrix(self, direction, anchor_attn):
        direction = normalize_text(direction) if direction else ""

        mass_x = anchor_attn.sum(dim=0)
        mass_y = anchor_attn.sum(dim=1)

        prob_x = mass_x / (mass_x.sum() + 1e-8)
        prob_y = mass_y / (mass_y.sum() + 1e-8)

        cdf_x = torch.cumsum(prob_x, dim=0)
        cdf_y = torch.cumsum(prob_y, dim=0)

        alpha = 15.0
        beta = 0.65
        center_beta = 0.5

        spatial_matrix = torch.ones((self.res, self.res)).to(anchor_attn.device)

        if "left" in direction:
            spatial_matrix *= torch.sigmoid(alpha * ((1.0 - cdf_x) - beta)).unsqueeze(0).expand(self.res, self.res)
        elif "right" in direction:
            spatial_matrix *= torch.sigmoid(alpha * (cdf_x - beta)).unsqueeze(0).expand(self.res, self.res)
        elif "middle" in direction or "center" in direction:
            spatial_matrix *= torch.sigmoid(
                alpha * ((1.0 - torch.abs(cdf_x - 0.5) * 2.0) - center_beta)
            ).unsqueeze(0).expand(self.res, self.res)

        if "top" in direction or "upper" in direction or "above" in direction:
            spatial_matrix *= torch.sigmoid(alpha * ((1.0 - cdf_y) - beta)).unsqueeze(1).expand(self.res, self.res)
        elif "bottom" in direction or "lower" in direction or "below" in direction:
            spatial_matrix *= torch.sigmoid(alpha * (cdf_y - beta)).unsqueeze(1).expand(self.res, self.res)

        return spatial_matrix

    def execute_graph(self, json_node):
        if not json_node or not isinstance(json_node, dict):
            return torch.ones((self.res, self.res)).to(self.cross_attn.device)

        target_info = json_node.get("target", json_node)
        target_words = target_info.get("compound_words", [])

        spatial_keywords = ["left", "right", "top", "bottom", "upper", "lower", "middle", "center"]
        raw_attrs = target_info.get("attributes", [])
        clean_attrs = [attr for attr in raw_attrs if normalize_text(attr) not in spatial_keywords]
        absolute_directions = [normalize_text(attr) for attr in raw_attrs if normalize_text(attr) in spatial_keywords]

        target_attn = self.execute_compound(target_words)
        target_attn = self.execute_attributes(target_attn, clean_attrs)

        relation = json_node.get("relation")

        if relation and isinstance(relation, dict):
            raw_direction = relation.get("direction", "")
            direction = " ".join([str(d) for d in raw_direction]) if isinstance(raw_direction, list) else str(raw_direction or "")

            anchor_node = relation.get("anchor")
            if isinstance(anchor_node, str):
                anchor_node = {"compound_words": [anchor_node], "attributes": []}
            elif isinstance(anchor_node, list):
                anchor_node = {"compound_words": [str(w) for w in anchor_node], "attributes": []}
            elif not isinstance(anchor_node, dict):
                anchor_node = None

            anchor_words = anchor_node.get("compound_words", []) if anchor_node else []
            anchor_attn = (
                self.execute_graph(anchor_node)
                if anchor_node
                else torch.ones((self.res, self.res)).to(self.cross_attn.device)
            )

            if direction:
                spatial_matrix = self.generate_spatial_matrix(direction, anchor_attn)
                target_attn = target_attn * spatial_matrix
            else:
                intersection = set([normalize_text(x) for x in target_words]) & set([normalize_text(x) for x in anchor_words])
                if intersection:
                    a_min, a_max = anchor_attn.min(), anchor_attn.max()
                    anchor_mask = (anchor_attn - a_min) / (a_max - a_min + 1e-8)
                    suppression_mask = 1.0 - anchor_mask
                    target_attn = torch.clamp(target_attn * suppression_mask, min=0.0)

        for abs_dir in absolute_directions:
            global_anchor = torch.ones((self.res, self.res)).to(self.cross_attn.device)
            target_attn = target_attn * self.generate_spatial_matrix(abs_dir, global_anchor)

        return target_attn
