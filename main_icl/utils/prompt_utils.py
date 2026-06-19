import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer


def calc_f1_score(pred, target, average="all"):
    """Calculate the F1 score of the prediction"""
    assert len(pred) == len(target), f"Length mismatch: {len(pred)} != {len(target)}"
    if average == "all":
        return {
            a: f1_score(target, pred, average=a) for a in ["micro", "macro", "weighted"]
        }
    return f1_score(target, pred, average=average)


def verbalize(verbalizer, samples, device):
    full_token, ans_mask, io_mask = verbalizer(samples, io_sep_mask=True)
    full_token = {k: v.to(device) for k, v in full_token.items()}
    ans_mask = ans_mask.to(device)
    io_mask = io_mask.to(device)
    return full_token, ans_mask, io_mask



def last_only_io_mask(io_mask):
    ones_indices = io_mask.squeeze().nonzero()
    last_one_index = ones_indices[-1].item()
    last_only_mask = torch.zeros_like(io_mask)
    last_only_mask[..., last_one_index] = True
    return last_only_mask


class Verbalizer:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        example_sep="\n\n",
        input_output_sep='',
        max_len_per_example=256,
        instruction="",
    ):
        """Seperate the input and output with #`io_sep_cnt` `sep`s
        and #`ex_sep_cnt` `sep`s between examples"""
        self.tokenizer = tokenizer
        self.ex_sep_ids = tokenizer(example_sep, add_special_tokens=False)["input_ids"]
        self.io_sep_ids = tokenizer(input_output_sep, add_special_tokens=False)[
            "input_ids"
        ]
        self.max_len = max_len_per_example
        self.max_example_len = (
            max_len_per_example - len(self.ex_sep_ids) - len(self.io_sep_ids) + 1
        )
        self.ex_sep_mark = "<ex_sep>"
        self.io_sep_mark = "<io_sep>"
        self.inst_tokens = self.tokenizer.tokenize(instruction)
        tokenizer.add_special_tokens(
            {"sep_token": x for x in (self.ex_sep_mark, self.io_sep_mark)}
        )

    def format_one_sample(
        self, sample: dict, no_output=False, max_opt_len=0
    ) -> list[str]:
        """Format one sample into a list of tokenized strings

        `max_opt_len`: length to reserve for the longest option"""
        max_len = self.max_example_len - max_opt_len
        assert (
            max_len > 0
        ), f"max example length is too short, unable to fit option len {max_opt_len}"
        untrunc_text = f"{sample['input'].rstrip()}{self.io_sep_mark}{'' if no_output else sample['output']}"
        full_list = self.tokenizer.tokenize(untrunc_text)
        return full_list

    def get_token_and_mask(self, full_list):
        """Convert to tokens and find where the answer begins"""
        last_io_sep = len(full_list) - full_list[::-1].index(self.io_sep_mark) - 1
        quest_list = full_list[: last_io_sep + 1]
        full_token, add_len = self.convert_text_to_ids(full_list, return_tensors=True)
        ans_mask = torch.zeros_like(full_token["input_ids"], dtype=torch.bool)
        ans_mask[:, len(quest_list) + add_len :] = True
        return full_token, ans_mask

    def convert_text_to_ids(self, text: list[str], return_tensors=False):
        """Convert list of text with marks to ids
        and calculate additional length due to special tokens"""
        vocab = self.tokenizer.get_vocab()
        ret = []
        additional_len = 0
        for t in text:
            if t == self.ex_sep_mark:
                ret += self.ex_sep_ids
                additional_len += len(self.ex_sep_ids) - 1
            elif t == self.io_sep_mark:
                ret += self.io_sep_ids
                additional_len += len(self.io_sep_ids) - 1
            else:
                ret.append(vocab[t])
        if return_tensors:
            ret = {
                "input_ids": torch.tensor(ret, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones((1, len(ret)), dtype=torch.long),
            }
        return ret, additional_len

    def get_intermediate_ans_mask(self, text: list[str]):
        """Find where all the <io_sep> are and build a mask for them"""
        m = []
        for t in text:
            if t == self.io_sep_mark:
                if len(self.io_sep_ids) >0:
                    m += [False] * (len(self.io_sep_ids) - 1)
                    m.append(True)
            elif t == self.ex_sep_mark:
                m += [False] * len(self.ex_sep_ids)
            else:
                m.append(False)
        m = torch.tensor(m, dtype=torch.bool).unsqueeze(0)
        return m

    def __call__(self, samples: list, all_options=False, io_sep_mask=False):
        """Format and tokenize the given samples, return the tokenized text and the answer mask"""
        full_list = self.inst_tokens.copy()
        if len(full_list):
            full_list.append(self.ex_sep_mark)
        for s in samples[:-1]:
            full_list += self.format_one_sample(s)
            full_list.append(self.ex_sep_mark)

        if all_options:
            options = [(o, self.tokenizer.tokenize(o)) for o in samples[-1]["options"]]
            opt_to_lens = {o: len(tokens) for o, tokens in options}
            max_opt_len = max(opt_to_lens.values())
            full_list += self.format_one_sample(
                samples[-1], no_output=True, max_opt_len=max_opt_len
            )
            ret = []
            for _, tokens in options:
                an_option = full_list + tokens
                full_token, ans_mask = self.get_token_and_mask(an_option)
                if io_sep_mask:
                    io_mask = self.get_intermediate_ans_mask(an_option)
                    ret.append((full_token, ans_mask, io_mask))
                else:
                    ret.append((full_token, ans_mask))
            return ret
        else:
            full_list += self.format_one_sample(samples[-1])
            full_token, ans_mask = self.get_token_and_mask(full_list)
            if io_sep_mask:
                io_mask = self.get_intermediate_ans_mask(full_list)
                return full_token, ans_mask, io_mask
            return full_token, ans_mask
