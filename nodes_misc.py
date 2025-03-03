
import folder_paths

import os
import random



class TextBox1:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                     "text1": ("STRING", {"default": "", "multiline": True}),
                    },
                     "optional": 
                    {
                    }  
               }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text1",)
    FUNCTION = "main"
    
    CATEGORY = "RES4LYF/text"
    DESCRIPTION = "Multiline textbox."

    def main(self, text1):

        return (text1,)

class TextBox3:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                     "text1": ("STRING", {"default": "", "multiline": True}),
                     "text2": ("STRING", {"default": "", "multiline": True}),
                     "text3": ("STRING", {"default": "", "multiline": True}),
                    },
                     "optional": 
                    {
                    }  
               }
    RETURN_TYPES = ("STRING", "STRING","STRING",)
    RETURN_NAMES = ("text1", "text2", "text3",)
    FUNCTION = "main"
    
    CATEGORY = "RES4LYF/text"
    DESCRIPTION = "Multiline textbox."

    def main(self, text1, text2, text3 ):

        return (text1, text2, text3, )



class TextLoadFile:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir)
                 if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith('.txt')]
        return {
            "required": {
                "text_file": (sorted(files), {"text_upload": True})
            }
        }
        

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/text"

    def main(self, text_file):
        input_dir = folder_paths.get_input_directory()
        text_file_path = os.path.join(input_dir, text_file) 
        if not os.path.exists(text_file_path):
            print(f"Error: The file `{text_file_path}` cannot be found.")
            return ("",)
        with open(text_file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return (text,)



class TextShuffle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":        ("STRING", {"forceInput": True}),
                "separator":   ("STRING", {"default": " ", "multiline": False}),
                "seed":        ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("shuffled_text",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/text"

    def main(self, text, separator, seed, ):
        if seed is not None:
            random.seed(seed)
        parts = text.split(separator)
        random.shuffle(parts)
        shuffled_text = separator.join(parts)

        return (shuffled_text, )



def truncate_tokens(text, truncate_to, clip, clip_type, stop_token):
    if truncate_to == 0:
        return ""
    
    truncate_words_to = truncate_to
    total = truncate_to + 1
    
    tokens = {}

    while total > truncate_to:
        words = text.split()
        truncated_words = words[:truncate_words_to]
        truncated_text = " ".join(truncated_words)

        try:
            tokens[clip_type] = clip.tokenize(truncated_text)[clip_type]
        except:
            return ""

        if clip_type not in tokens:
            return truncated_text

        clip_end=0
        for b in range(len(tokens[clip_type])):
            for i in range(len(tokens[clip_type][b])):
                clip_end += 1
                if tokens[clip_type][b][i][0] == stop_token:
                    break
        if clip_type == 'l' or clip_type == 'g':
            clip_end -= 2
        elif clip_type == 't5xxl':
            clip_end -= 1

        total = clip_end

        truncate_words_to -= 1
        
    return truncated_text



class TextShuffleAndTruncate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":               ("STRING", {"forceInput": True}),
                "separator":          ("STRING", {"default": " ", "multiline": False}),
                "truncate_words_to":  ("INT", {"default": 77, "min": 1, "max": 10000}),
                "truncate_tokens_to": ("INT", {"default": 77, "min": 1, "max": 10000}),
                "seed":               ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "clip": ("CLIP", ),
            }
        }

    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING",)
    RETURN_NAMES = ("shuffled_text", "text_words","text_clip_l","text_clip_g","text_t5",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/text"

    def main(self, text, separator, truncate_words_to, truncate_tokens_to, seed, clip=None):
        if seed is not None:
            random.seed(seed)
        parts = text.split(separator)
        random.shuffle(parts)
        shuffled_text = separator.join(parts)

        words = shuffled_text.split()
        truncated_words = words[:truncate_words_to]
        truncated_text = " ".join(truncated_words)
        
        t5_name = "t5xxl" if not hasattr(clip.tokenizer, "pile_t5xl") else "pile_t5xl"

        text_clip_l = truncate_tokens(truncated_text, truncate_tokens_to, clip, "l",     49407)
        text_clip_g = truncate_tokens(truncated_text, truncate_tokens_to, clip, "g",     49407)
        text_t5     = truncate_tokens(truncated_text, truncate_tokens_to, clip, t5_name, 1)
                    
        return (shuffled_text, truncated_text, text_clip_l, text_clip_g, text_t5,)



class TextTruncateTokens:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":               ("STRING", {"forceInput": True}),
                "truncate_words_to":  ("INT", {"default": 30, "min": 0, "max": 10000}),
                "truncate_clip_l_to": ("INT", {"default": 77, "min": 0, "max": 10000}),
                "truncate_clip_g_to": ("INT", {"default": 77, "min": 0, "max": 10000}),
                "truncate_t5_to":     ("INT", {"default": 77, "min": 0, "max": 10000}),
            },
            "optional": {
                "clip":               ("CLIP", ),
            }
        }

    RETURN_TYPES = ("STRING","STRING","STRING","STRING",)
    RETURN_NAMES = ("text_words","text_clip_l","text_clip_g","text_t5",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/text"

    def main(self, text, truncate_words_to, truncate_clip_l_to, truncate_clip_g_to, truncate_t5_to, clip=None):

        words = text.split()
        truncated_words = words[:truncate_words_to]
        truncated_text = " ".join(truncated_words)
        
        t5_name = "t5xxl" if not hasattr(clip.tokenizer, "pile_t5xl") else "pile_t5xl"

        if clip is not None:
            text_clip_l = truncate_tokens(text, truncate_clip_l_to, clip, "l",     49407)
            text_clip_g = truncate_tokens(text, truncate_clip_g_to, clip, "g",     49407)
            text_t5     = truncate_tokens(truncated_text, truncate_t5_to, clip, t5_name, 1)
        else:
            text_clip_l = None
            text_clip_g = None
            text_t5     = None
        
        return (truncated_text, text_clip_l, text_clip_g, text_t5,)



class TextConcatenate:

    @ classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                },
                "optional": {
                "text_1":    ("STRING", {"multiline": False, "default": "", "forceInput": True}),                
                "text_2":    ("STRING", {"multiline": False, "default": "", "forceInput": True}), 
                "separator": ("STRING", {"multiline": False, "default": ""}),                
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/text"

    def main(self, text_1="", text_2="", separator=""):
    
        return (text_1 + separator + text_2, )



class TextBoxConcatenate:

    @ classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                 "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "text_external": ("STRING", {"multiline": False, "default": "", "forceInput": True}),                
                "separator":     ("STRING", {"multiline": False, "default": ""}),        
                "mode":          (['append_external_input', 'prepend_external_input',],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/text"
    DESCRIPTION = "Multiline textbox with concatenate functionality."


    def main(self, text="", text_external="", separator="", mode="append_external_input"):
        if   mode == "append_external_input":
            text = text + separator + text_external
        elif mode == "prepend_external_input":
            text = text_external + separator + text
    
        return (text, )



class SeedGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("INT",  "INT",)
    RETURN_NAMES = ("seed", "seed+1",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/utilities"

    def main(self, seed,):
        return (seed, seed+1,)

