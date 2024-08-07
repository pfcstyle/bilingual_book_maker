@echo off
set SCRIPT_DIR=%~dp0

call "%SCRIPT_DIR%\env\Scripts\activate.bat"

python make_book.py --book_name test_books/The_Digital_Factory_The_Human_Labor.pdf --model tencentransmart --prompt prompt_template_sample.json --test REM--resume --single_translate
