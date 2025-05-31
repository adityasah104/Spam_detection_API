# feature_extraction.py
import re
import unicodedata
import string
from collections import Counter
import numpy as np

class SMSFeatureExtractor:
    def __init__(self):
        self.feature_columns = [
            'has_phone_number', 'has_special_chars', 'has_all_caps_words',
            'has_url', 'has_short_url', 'has_regular_url', 'is_mixed_language', 
            'has_currency', 'date', 'time', 'has_id_code', 'has_emoji',
            'has_repeated_words', 'has_consecutive_special_chars',
            'has_subscriber_code', 'avg_word_length', 'word_length'
        ]
    
    def extract_phone_numbers(self, text):
        """Enhanced phone number extraction with comprehensive patterns"""
        if not isinstance(text, str):
            return 0
        
        digit_map = {
            '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
            '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
        }
        
        converted_text = text
        for bn_digit, ar_digit in digit_map.items():
            converted_text = converted_text.replace(bn_digit, ar_digit)
        
        phone_patterns = [
            r'\+?\d{2}\s*\d{10}',
            r'\d{10,11}',
            r'\+?\d{1,4}[-\s]?\d{2,4}[-\s]?\d{2,4}[-\s]?\d{2,4}',
            r'\b\d{5,9}\b',
            r'\b\d{2,5}[-\s]\d{3,5}\b',
            r'\(\d{2,4}\)[-\s]?\d{3,6}\b',
            r'\b0\d{2,4}[-\s]?\d{3,6}\b',
            r'\d{3,4}\s+\d{2}\s+\d{5}',
            r'\(\d{3,4}\)\s*\d{6,8}'
        ]
        
        mixed_patterns = [
            r'\+?[০-৯\d]{2}\s*[০-৯\d]{10}',
            r'[০-৯\d]{3,5}[-\s]?[০-৯\d]{6,7}',
            r'[০-৯\d]{10,11}',
            r'\b[০-৯\d]{5,9}\b',
            r'\b[০-৯\d]{2,5}[-\s][০-৯\d]{3,5}\b'
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text) or re.search(pattern, converted_text)
            if match:
                clean_match = re.sub(r'[-\s\(\)]', '', match.group())
                if len(clean_match) >= 5:
                    return 1
        
        for pattern in mixed_patterns:
            match = re.search(pattern, text)
            if match:
                clean_match = match.group()
                for bn_digit, ar_digit in digit_map.items():
                    clean_match = clean_match.replace(bn_digit, ar_digit)
                clean_match = re.sub(r'\D', '', clean_match)
                if len(clean_match) >= 5:
                    return 1
        
        return 0
    
    def extract_special_chars(self, text):
        """Enhanced special character detection excluding URLs"""
        if not isinstance(text, str):
            return 0
        
        url_pattern = r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*'
        text_without_urls = re.sub(url_pattern, '', text)
        
        special_chars = set(string.punctuation)
        special_chars.discard('₹')
        special_chars.discard('?')
        special_chars.discard(',')
        special_chars.discard('.')
        
        return 1 if any(char in special_chars for char in text_without_urls) else 0
    
    def extract_all_caps_words(self, text):
        """Enhanced all caps detection for Latin script only"""
        if not isinstance(text, str):
            return 0
        
        latin_words = re.findall(r'[A-Za-z]+', text)
        return 1 if latin_words and any(word.isupper() and len(word) > 1 for word in latin_words) else 0
    
    def extract_urls(self, text):
        """Enhanced URL detection with comprehensive patterns and detailed classification"""
        if not isinstance(text, str):
            return {'has_url': 0, 'has_short_url': 0, 'has_regular_url': 0}
        
        regular_url_patterns = [
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
            r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        ]
        
        short_url_patterns = [
            r'bit\.ly/\S+', 
            r'goo\.gl/\S+', 
            r'tinyurl\.com/\S+',
            r't\.co/\S+'
        ]
        
        has_short_url = 0
        has_regular_url = 0
        
        for pattern in short_url_patterns:
            if re.search(pattern, text):
                has_short_url = 1
                break
        
        for pattern in regular_url_patterns:
            if re.search(pattern, text):
                match = re.search(pattern, text)
                matched_text = match.group(0)
                
                is_short_url = 0
                for short_pattern in short_url_patterns:
                    if re.search(short_pattern, matched_text):
                        is_short_url = 1
                        break
                
                if not is_short_url:
                    has_regular_url = 1
                break
        
        has_url = 1 if (has_short_url or has_regular_url) else 0
        
        return {
            'has_url': has_url,
            'has_short_url': has_short_url,
            'has_regular_url': has_regular_url
        }
    
    def extract_mixed_language(self, text):
        """Enhanced mixed language detection (Bengali/Assamese + Latin)"""
        if not isinstance(text, str):
            return 0
        
        url_pattern = r'https?://\S+|www\.\S+'
        text_without_urls = re.sub(url_pattern, '', text)
        
        has_bengali_assamese = 0
        has_latin = bool(re.search(r'[A-Za-z]', text_without_urls))
        
        for char in text_without_urls:
            try:
                char_name = unicodedata.name(char, '')
                if 'BENGALI' in char_name or 'ASSAMESE' in char_name:
                    has_bengali_assamese = 1
                    break
            except ValueError:
                continue
        
        return 1 if has_bengali_assamese and has_latin else 0
    
    def extract_currency(self, text):
        """Enhanced currency detection with comprehensive patterns"""
        if not isinstance(text, str):
            return 0
        
        simple_patterns = [
            r'₹', r'रुपया', 
            r'Rs\.?', r'INR',
            r'\$', r'€', r'£', r'¥',
            r'dollar', r'euro', r'rupee'
        ]
        
        for pattern in simple_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1
        
        currency_terms = [
            'টাকা', 'টকা', 'পয়সা', 'পোইসা', 'তাকা',
            'টকা', 'টাকা', 'পইচা', 'পাই', 'ধন',
            'taka', 'toka', 'poisa', 'paisa', 'টকীয়া'
        ]
        
        for term in currency_terms:
            if term in text:
                return 1
        
        for term in currency_terms:
            pattern = term + r'\w*'
            if re.search(pattern, text):
                return 1
        
        return 0
    
    def extract_date(self, text):
        """Check for date/time patterns"""
        if not isinstance(text, str):
            return 0
        
        date_patterns = [
            r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}',
            r'\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}',
            r'\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?',
            r'\d{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর)(?:\w{0,3})?(?:,)?\s+\d{2,4}',
            r'\d{1,2}\s+(?:জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)(?:\w{0,3})?(?:,)?\s+\d{2,4}',
            r'(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর)(?:\w{0,3})?\s+\d{1,2}(?:,)?\s+\d{2,4}',
            r'(?:জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)(?:\w{0,3})?\s+\d{1,2}(?:,)?\s+\d{2,4}',
            r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,)?\s+\d{2,4}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:,)?\s+\d{2,4}',
            r'\d{4}\s*সাল',
            r'\d{4}\s*(?:year|বছর|বৰ্ষ)',
            r'[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)(?:,)?\s+[১২৩৪৫৬৭৮৯০]{2,4}',
            r'[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)(?:ৰ)?\s+আগত',
            r'\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+তাৰিখে',
            r'\d{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)(?:,)?\s+\d{2,4}\s+(?:তারিখ(?:ে|ের|)|তাৰিখ(?:ে|ৰ|ত))',
            r'[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)(?:,)?\s+[১২৩৪৫৬৭৮৯০]{2,4}\s+(?:তারিখ(?:ে|ের|)|তাৰিখ(?:ে|ৰ|ত))',
            r'(?:আজি|আজ|কালি|কাল|গতকালি|গতকাল|পরশু|পৰহি|যোৱা)\s+[১২৩৪৫৬৭৮৯০]{1,2}(?:ই|ৰ|র)?\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|نভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)',
            r'[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)(?:,)?\s+[১২৩৪৫৬৭৮৯০]{2,4}\s+দিন(?:ত|ে)?',
            r'(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱار(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)\s+মা(?:স|হ)',
            r'(?:আজি|আজ|কালি|কাল|গতকালি|গতকাল|পরশু|পৰহি)\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)(?:,)?\s+[১২৩৪৫৬৭৮৯০]{1,4}',
            r'[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)(?:ত|[[:punct:]])?',
            r'[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ).*?[১২৩৪৫৬৭৮৯০]{4}',
            r'[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ).*?দিৱস.*?[১২৩৪৫৬৭৮৯০]{4}',
            r'[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)\s+(?:".*?"|\'.*?\').*?[১২৩৪৫৬৭৮৯০]{4}',
        ]
        context_patterns = [
            r'(?:আজি|আজ|কালি|কাল|গতকালি|গতকাল|পরশু|পৰহি|যোৱা)\s+[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:দিন|দিনত)',
            r'(?:গত|বিগত|যোৱা|আহিবলগীয়া)\s+(?:সপ্তাহ|সপ্তাহত|সোমবাৰ|মঙ্গলবাৰ|বুধবাৰ|বৃহস্পতিবাৰ|শুক্ৰবাৰ|শনিবাৰ|ৰবিবাৰ|সোমবার|মঙ্গলবার|বুধবার|বৃহস্পতিবার|শুক্রবার|শনিবার|রবিবার)',
            r'(?:সোমবাৰ|মঙ্গলবাৰ|বুধবাৰ|বৃহস্পতিবাৰ|শুক্ৰবাৰ|শনিবাৰ|ৰবিবাৰ|সোমবার|মঙ্গলবার|বুধবার|বৃহস্পতিবার|শুক্রবার|শনিবার|রবিবার)\s+[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)',
            r'উৎসৱমুখৰ\s+ছুটিৰ\s+দিনটো',
            r'অফাৰ\s+শেষ\s+হ\'ব\s+[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)(?:ত|[[:punct:]])?',
            r'(?:আগত|পূৰ্বে)\s+এমাহৰ\s+বাবে', 
            r'[১২৩৪৫৬৭৮৯০]{1,2}\s+(?:জান(?:ু|ূ)য়ার(?:ী|ি)|ফেব্র(?:ু|ূ)য়ার(?:ী|ি)|মার্চ|এপ্রিল|মে|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|জান(?:ু|ূ)ৱার(?:ী|ি)|ফেব্র(?:ু|ূ)ৱার(?:ী|ি)|মাৰ্চ|এপ্ৰিল|জ(?:ু|ূ)ন|জ(?:ু|ূ)লাই|আগষ্ট|ছেপ্টেম্বৰ|অক্টোবৰ|নৱেম্বৰ|ডিচেম্বৰ)(?:ৰ)?\s+(?:আগত|পূৰ্বে)',
        ]       
        all_patterns = date_patterns + context_patterns
        
        for pattern in all_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1
        
        return 0
    
    def extract_time(self, text):
        """Enhanced time detection with multiple patterns"""
        if not isinstance(text, str):
            return 0
        
        english_patterns = [
            r'\b([0-9]|0[0-9]|1[0-2])(?::([0-5][0-9]))?(?::([0-5][0-9]))?\s*([AaPp][Mm])\b',
            r'\b([0-9]|0[0-9]|1[0-2])(?:\.|\s)([0-5][0-9])(?:\s*|\.)([AaPp]\.?[Mm]\.?)\b',
            r'\b([01]?[0-9]|2[0-3]):([0-5][0-9])(?::([0-5][0-9]))?\b',
            r'\b(noon|midnight|midday)\b',
            r'\b([0-9]|0[0-9]|1[0-2])\s+o\'?clock\b',
            r'\b(half|quarter)\s+(past|to)\s+([0-9]|0[0-9]|1[0-2])\b',
            r'\bat\s+([0-9]|0[0-9]|1[0-2])(?:\s+|\:)([0-5][0-9])?\s*([AaPp][Mm])?\b'
        ]
        
        bn_as_numeric_patterns = [
            r'[০-৯]{1,2}(?:[:\.।]|\s*ঃ|\s+)[০-৯]{1,2}(?:[:\.।]|\s*ঃ|\s+)?[০-৯]{0,2}',
            r'[০-৯]{1,2}\s*(?:টা|বাজে|ঘণ্টা|घंटा)'
        ]
        
        bn_as_time_words = [
            'সকাল', 'ভোর', 'রাত', 'বিকাল', 'সন্ধ্যা', 'দুপুর',
            'ৰাতি', 'পুৱা', 'গধূলি', 'আবেলি', 'নিশা', 'দুপৰীয়া',
            'টা', 'বাজে', 'ঘণ্টা', 'মিনিট', 'সেকেন্ড',
            'বজি', 'বাজি', 'ঘণ্টা', 'মিনিট', 'ছেকেণ্ড'
        ]
        
        time_phrases = [
            r'(?:এখন|বর্তমান|এতিয়া)\s+[০-৯]{1,2}\s*(?:টা|বাজে|ঘণ্টা|বজি)',
            r'(?:সকাল|ভোর|রাত|বিকাল|সন্ধ্যা|দুপুর|ৰাতি|পুৱা|গধূলি|আবেলি|নিশা|দুপৰীয়া)\s+[০-৯]{1,2}',
            r'[০-৯]{1,2}\s*(?:টা|বাজে|ঘণ্টা)\s*(?:ও|আৰু|এবং)?\s*[০-৯]{1,2}\s*(?:মিনিট|মিনিট)',
            r'(?:ভোরে|ভোৰত|সকালে|সকালত|রাতে|ৰাতিত)\s+[০-৯]{1,2}',
            r'[০-৯]{1,2}(?:[:\.।]|\s*ঃ|\s+)[০-৯]{1,2}\s*(?:এএম|পিএম|am|pm|a\.m\.|p\.m\.)'
        ]
        
        for pattern in english_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1
        
        for pattern in bn_as_numeric_patterns:
            if re.search(pattern, text):
                return 1
        
        if any(re.search(r'[০-৯]{1,2}\s*' + word, text) for word in bn_as_time_words):
            return 1
        
        for pattern in time_phrases:
            if re.search(pattern, text, re.IGNORECASE):
                return 1
        
        return 0
        
    def extract_id_codes(self, text):
        """Enhanced ID code detection with multiple patterns"""
        if not isinstance(text, str):
            return 0
        
        id_pattern1 = r'\b\w*\d{10,}\w*\b'
        id_pattern2 = r'\b[A-Za-z]+\d+\.\d+\.\d+\b'
        
        if re.search(id_pattern1, text) or re.search(id_pattern2, text):
            return 1
        
        return 0
    
    def extract_emojis(self, text):
        """Enhanced emoji detection with comprehensive Unicode ranges"""
        if not isinstance(text, str):
            return 0
        
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0000257F"
            "\U00002600-\U000026FF"
            "\U00002700-\U000027BF"
            "\U0000FE0F"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        
        return 1 if emoji_pattern.search(text) else 0
    
    def has_repeated_words(self, text):
        """Check for repeated words"""
        if not isinstance(text, str):
            return 0
        
        text = re.sub(r'[^\w\s\u0980-\u09FF\u0985-\u09FB]', '', text.lower())
        words = text.split()
        word_counts = Counter(words)
        
        return 1 if any(count > 1 for count in word_counts.values()) else 0
    
    def has_consecutive_special_chars(self, text):
        """Check for consecutive special characters"""
        if not isinstance(text, str):
            return 0
        
        url_pattern = r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*'
        text_without_urls = re.sub(url_pattern, '', text)
        
        special_chars_pattern = r'([\?\!\@\#\$\%\&\*\(\)\-\_\=\+\[\]\{\}\;\:\,\.\<\>\/\\\|])\1+'
        
        return 1 if re.search(special_chars_pattern, text_without_urls) else 0
    
    def detect_subscriber_codes(self, text):
        """Check for subscriber codes"""
        if not isinstance(text, str):
            return 0
        
        parentheses_pattern = r'\(([^\)]*)\)'
        parentheses_matches = re.findall(parentheses_pattern, text)
        
        for match in parentheses_matches:
            if re.search(r'\d{4,5}', match) or re.search(r'[*#]', match):
                return 1
        
        if re.search(r'\b\d{4,5}\b', text):
            return 1
        if re.search(r'\*\d+(?:\*\d+)*\#', text):
            return 1
        if re.search(r'\#\d+(?:\*\d+)*\#?', text):
            return 1
        if re.search(r'\b[A-Za-z]+\s+\d+\b', text):
            return 1
        
        return 0
    
    def calculate_avg_word_length(self, text):
        """Calculate average word length"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0
        
        text = re.sub(r'[^\w\s\u0980-\u09FF\u0985-\u09FB]', '', text)
        words = [word for word in text.split() if len(word) > 0]
        
        if not words:
            return 0.0
        
        total_length = sum(len(word) for word in words)
        return round(total_length / len(words), 2)

    def count_chars_without_spaces(self, text):
        """Count characters excluding spaces"""
        if not isinstance(text, str):
            return 0
        
        return sum(1 for char in text if char != ' ')

    def extract_features(self, text):
        """Extract all features and return as numpy array"""
        if not isinstance(text, str):
            text = str(text)
        
        # Extract URL features first since other features exclude URLs
        url_features = self.extract_urls(text)
        
        # Extract all features
        features = [
            self.extract_phone_numbers(text),
            self.extract_special_chars(text),
            self.extract_all_caps_words(text),
            url_features['has_url'],
            url_features['has_short_url'],
            url_features['has_regular_url'],
            self.extract_mixed_language(text),
            self.extract_currency(text),
            self.extract_date(text),
            self.extract_time(text),
            self.extract_id_codes(text),
            self.extract_emojis(text),
            self.has_repeated_words(text),
            self.has_consecutive_special_chars(text),
            self.detect_subscriber_codes(text),
            self.calculate_avg_word_length(text),
            self.count_chars_without_spaces(text)
        ]
        
        return np.array(features).reshape(1, -1)

# Example usage
if __name__ == "__main__":
    extractor = SMSFeatureExtractor()
    