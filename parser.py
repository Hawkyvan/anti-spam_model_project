import os
import pandas as pd
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import chardet

# Загрузка необходимых ресурсов NLTK
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

def preproc_text(text):
    '''Очищает текст: удаляет HTML, URL, небуквенные символы, приводит к нижнему регистру,
    удаляет стоп-слова и применяет лемматизацию.
    '''
    if not isinstance(text, str) or not text.strip():
        return ""

    # Удаление HTML-тегов
    try:
        text = BeautifulSoup(text, 'html.parser').get_text(separator=' ')
    except Exception as e:
        print(f'Ошибка обработки HTML: {e}')
        return ''

    # Удаление URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Удаление небуквенных символов и чисел
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Приведение к нижнему регистру
    text = text.lower()

    # Токенизация
    words = word_tokenize(text)

    # Удаление стоп-слов
    stop_words_en = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words_en]

    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return ' '.join(lemmatized_words)


def detect_encoding(content):
    '''Определяет кодировку содержимого, если она не указана в письме.'''
    
    result = chardet.detect(content)
    return result['encoding'] if result['encoding'] else 'utf-8'


def decode(content, charset):
    '''Декодирование содержимого письма с проверкой кодировки.'''
    
    if charset is None or charset.lower() in ['', 'default', 'default_charset', 'unknown-8bit', 'charset=', 'iso-8410-6', 'iso-18899997-1']:
        charset = 'utf-8'  # Принудительно ставим UTF-8 при неизвестной кодировке

    try:
        return content.decode(charset, errors='ignore')
    except (LookupError, TypeError, ValueError):
        print(f'[!] Ошибка: неизвестная кодировка {charset}, пробуем auto-detect')
        return content.decode(detect_encoding(content), errors='ignore')


def parse_email(file_path):
    '''Парсинг email-файла .eml.'''
    
    try:
        with open(file_path, 'rb') as f:
            message = BytesParser(policy=policy.default).parse(f)
            
        data = {
            'From': message['From'],
            'To': message['To'],
            'Date': message['Date'],
            'Text': '',
            'Mark': ''
        }
        
        text_parts = []

        if message.is_multipart():
            for part in message.walk():
                if part.is_multipart():
                    continue  # Пропускаем контейнерные части
                
                content_type = part.get_content_type()
                content_disposition = part.get_content_disposition()

                if content_disposition and 'attachment' in content_disposition:
                    continue  # Пропускаем вложения
                
                try:
                    content = part.get_payload(decode=True)
                    if content is None:
                        continue  # Пропускаем пустые части

                    charset = part.get_content_charset()
                    decoded_content = decode(content, charset)  # Декодирование

                    if content_type == 'text/plain' or content_type == 'text/html':
                        text_parts.append(decoded_content)
                        
                except Exception as e:
                    print(f'Ошибка при обработке части письма {file_path}: {e}')
        else:
            try:
                content = message.get_payload(decode=True)
                if content is not None:
                    charset = message.get_content_charset() or detect_encoding(content)
                    decoded_content = decode(content, charset)

                    if message.get_content_type() == 'text/plain' or message.get_content_type() == 'text/html':
                        text_parts.append(decoded_content)

            except Exception as e:
                print(f'Ошибка при обработке письма {file_path}: {e}')

        # Объединяем извлеченный текст
        data['Text'] = ' '.join(text_parts).strip()

        # Применяем предобработку текста
        if data['Text']:
            data['Text'] = preproc_text(data['Text'])
        else:
            data['Text'] = ''
            
        # Определение спама
        if spam_dir == os.path.dirname(file_path):
            data['Mark'] = 'spam'
        elif ham_dir == os.path.dirname(file_path):
            data['Mark'] = 'not spam'
            
        return data
    except Exception as e:
        print(f'Ошибка при обработке файла {file_path}: {e}')
        return None


def parse_dir(directory):
    '''Парсинг всех .eml файлов в директории.'''
    
    eml_list = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.eml'):
            file_path = os.path.join(directory, filename)
            data = parse_email(file_path)
            if data is not None:
                eml_list.append(data)
    return eml_list


def save_csv(eml_list, out_csv):
    '''Сохранение списка email данных в CSV файл.'''
   
    df = pd.DataFrame(eml_list)
    df.to_csv(out_csv, sep=';')

# Пути к папкам с письмами
spam_dir = 'data/spam'
ham_dir = 'data/ham'
out_csv = 'data/eml_dataset.csv'

# Запуск парсинга и сохранения
eml_list = parse_dir(spam_dir)
eml_list.extend(parse_dir(ham_dir))
save_csv(eml_list, out_csv)

print("Парсинг завершен. Данные сохранены в", out_csv)
