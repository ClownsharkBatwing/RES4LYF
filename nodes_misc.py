
import torch


    
class TextBox1:
    @classmethod
    def INPUT_TYPES(s):
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
    
    CATEGORY = "res4lyf/text"
    DESCRIPTION = "Multiline textbox."

    def main(self, text1):

        return (text1,)

class TextBox3:
    @classmethod
    def INPUT_TYPES(s):
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
    
    CATEGORY = "res4lyf/text"
    DESCRIPTION = "Multiline textbox."

    def main(self, text1, text2, text3 ):

        return (text1, text2, text3, )

