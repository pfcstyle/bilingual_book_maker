#!/bin/sh
source ./env/bin/activate
python make_book.py --book_name test_books/test1.pdf --model tencentransmart --prompt prompt_template_sample.json --accumulated_num 1600 --test #--resume
