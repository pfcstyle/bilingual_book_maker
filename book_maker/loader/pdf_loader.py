import os
import re
import pickle
import traceback
import string
import sys
from copy import copy
from pathlib import Path

import fitz
from bs4 import BeautifulSoup as bs
from bs4 import Tag
from bs4.element import NavigableString
from ebooklib import ITEM_DOCUMENT, epub
from rich import print
from tqdm import tqdm

from book_maker.utils import num_tokens_from_text, prompt_config_to_kwargs

from .base_loader import BaseBookLoader
from .helper import EPUBBookLoaderHelper, is_text_link, not_trans


class PDFBookLoader(BaseBookLoader):
    def __init__(
        self,
        pdf_name,
        model,
        key,
        resume,
        src_language,
        language,
        model_api_base=None,
        is_test=False,
        test_num=5,
        prompt_config=None,
        single_translate=False,
        context_flag=False,
        temperature=1.0,
    ):
        self.pdf_name = pdf_name
        self.new_pdf = fitz.open()
        self.src_language = src_language
        self.translate_model = model(
            key,
            language,
            api_base=model_api_base,
            context_flag=context_flag,
            temperature=temperature,
            **prompt_config_to_kwargs(prompt_config),
        )
        self.is_test = is_test
        self.test_num = test_num
        self.translate_tags = "p"
        self.exclude_translate_tags = "sup"
        self.allow_navigable_strings = False
        self.accumulated_num = 1
        self.translation_style = ""
        self.context_flag = context_flag
        self.helper = EPUBBookLoaderHelper(
            self.translate_model,
            self.accumulated_num,
            self.translation_style,
            self.context_flag,
        )
        self.retranslate = None
        self.exclude_filelist = ""
        self.only_filelist = ""
        self.single_translate = single_translate
        self.block_size = -1
        self.origin_book = fitz.open(self.pdf_name)
        self.page_rect = self.origin_book.load_page(0).rect
        self.margins = {
            "left_margin": 50,
            "top_margin": 50,
            "right_margin": 50,
            "bottom_margin": 50,
        } # left, top, right, bottom

        self.p_to_save = []
        self.resume = resume
        self.bin_path = f"{Path(pdf_name).parent}/.{Path(pdf_name).stem}.temp.bin"
        if self.resume:
            self.load_state()

    @staticmethod
    def _is_special_text(text):
        return (
            text.isdigit()
            or text.isspace()
            or is_text_link(text)
            or all(char in string.punctuation for char in text)
        )

    def _make_new_book(self, book):
        new_book = fitz.open()
        return new_book
    
    def _extract_text_block_from_pdf(self):
        text_blocks = []
        for page_num in range(len(self.origin_book)):
            page = self.origin_book.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            prev_bbox = None
            for block in blocks:
                if block['type'] == 0:  # text block
                    for line in block["lines"]:
                        bbox = line["bbox"]
                        # 判断是否是新的段落
                        if text_blocks and prev_bbox and (bbox[0] - prev_bbox[0] > 5 or bbox[2] - prev_bbox[2] > 5):
                            # 在上一个text_block的text末尾添加两个换行符
                            text_blocks[-1]["text"] += "\n\n"
                        prev_bbox = bbox
                        for span in line["spans"]:
                            new_text: str = span["text"]
                            if new_text.strip() == "":
                                continue
                            last_text_block = text_blocks[-1] if text_blocks else None
                            if last_text_block and last_text_block["font"] == span["font"] and last_text_block["font_size"] == span["size"]:
                                last_text_block["text"] += new_text
                            else:
                                
                                if '\n' in new_text:
                                    print(new_text)
                                if len(text_blocks) > 200:
                                    print("too many text blocks")
                                    return text_blocks
                                text_blocks.append({
                                    "text": new_text,
                                    "font_size": span["size"],
                                    "font": span["font"],
                                    "flags": span["flags"],
                                    "page": page_num
                                })
        return text_blocks

    
    def _split_into_paragraphs(self, text):
        # 保留段落符
        parts = re.split(r'(\n\n)', text)
        paragraphs = []

        current_paragraph = ""
        for part in parts:
            if part == '\n\n':
                current_paragraph += part
                paragraphs.append(current_paragraph)
                current_paragraph = ""
            else:
                current_paragraph += part
        
        # 添加最后的段落，如果有的话
        if current_paragraph:
            paragraphs.append(current_paragraph)

        return paragraphs

    def _split_long_paragraph(self, paragraph, lang: str, max_length=1600):
        if len(paragraph) <= max_length:
            return [paragraph]

        if lang.startswith('zh'):
            sentences = re.split(r'([。！？])', paragraph)
        elif lang == 'en':
            sentences = re.split(r'([.!?])', paragraph)
        else:
            sentences = re.split(r'([.!?。！？])', paragraph)

        new_paragraphs = []
        current_paragraph = ""

        # sentences 列表中的偶数索引包含句子，奇数索引包含标点符号
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # 添加标点符号

            if len(current_paragraph) + len(sentence) > max_length:
                new_paragraphs.append(current_paragraph)
                current_paragraph = sentence
            else:
                current_paragraph += sentence

        if current_paragraph:
            new_paragraphs.append(current_paragraph)

        return new_paragraphs
    
    def _extract_paragraphs(self):
        text_blocks = self._extract_text_block_from_pdf()
        new_text_blocks = []
        for block in text_blocks:
            text = block['text']
            if not text:
                continue

            paragraphs = self._split_into_paragraphs(text)
            for paragraph in paragraphs:
                short_paragraphs = self._split_long_paragraph(paragraph, self.src_language, 1600 if self.accumulated_num == 1 else self.accumulated_num)
                for short_paragraph in short_paragraphs:
                    new_text_block = block.copy()
                    new_text_block['text'] = short_paragraph
                    new_text_blocks.append(new_text_block)
        return new_text_blocks

    def _split_text_into_lines(self, text, font, font_size, max_width):
        lines = []
        current_line = ""
        for word in text:
            test_line = current_line + word
            if fitz.get_text_length(test_line, fontsize=font_size) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines
    
    def _save_text_to_pdf(self, text_block, line_spacing=2):
        document = self.new_pdf
        page_size = self.page_rect
        margins = self.margins
        text = text_block["text"]
        font_name = text_block["font"]
        font_size = text_block["font_size"]

        max_width = page_size.width - margins["left_margin"] - margins["right_margin"]
        lines = self._split_text_into_lines(text, font_name, font_size, max_width)
        
        current_page = None
        current_y = margins["top_margin"]

        for line in lines:
            if current_page is None or current_y + font_size + line_spacing > page_size.height - margins["bottom_margin"]:
                current_page = document.new_page(width=page_size.width, height=page_size.height)
                current_y = margins["top_margin"]

            text_rect = fitz.Rect(margins["left_margin"], current_y, page_size.width - margins["right_margin"], current_y + font_size + line_spacing)
            current_page.insert_textbox(text_rect, line, fontsize=font_size)
            current_y += font_size + line_spacing

    def make_bilingual_book(self):
        all_paragraphs_blocks = self._extract_paragraphs()
        all_p_length = len(all_paragraphs_blocks)
        pbar = tqdm(total=self.test_num) if self.is_test else tqdm(total=all_p_length)
        index = 0
        p_to_save_len = len(self.p_to_save)
        try:
            index = 0
            for p_block in all_paragraphs_blocks:
                try:
                    self._save_text_to_pdf(p_block)
                    if index < p_to_save_len and self.resume:
                        p_block['text'] = self.p_to_save[index]
                    else:
                        translated_text = self.translate_model.translate(p_block['text'])
                        p_block['text'] = translated_text
                        self.p_to_save.append(translated_text)
                        if index % 20 == 0:
                            self._save_progress()
                    self._save_text_to_pdf(p_block)
                    index += 1
                    pbar.update(1)
                    if self.is_test and index >= self.test_num:
                        break
                except Exception as e:
                    print(e)
                    raise Exception("Something is wrong when translate") from e
                
            
            name, _ = os.path.splitext(self.pdf_name)
            self.new_pdf.save(f"{name}_bilingual.pdf")
            pbar.close()
        except (KeyboardInterrupt, Exception) as e:
            traceback.print_exc()

            print("you can resume it next time")
            self._save_progress()
            self._save_temp_book()
            sys.exit(0)

    def load_state(self):
        try:
            with open(self.bin_path, "rb") as f:
                self.p_to_save = pickle.load(f)
        except Exception:
            raise Exception("can not load resume file")

    def _save_temp_book(self):
        name, _ = os.path.splitext(self.pdf_name)
        self.new_pdf.save(f"{name}_bilingual_temp.pdf")

    def _save_progress(self):
        try:
            with open(self.bin_path, "wb") as f:
                pickle.dump(self.p_to_save, f)
        except Exception:
            raise Exception("can not save resume file")
