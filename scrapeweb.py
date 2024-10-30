import requests
import tldextract
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urlparse
import datetime
import os  # Import os to check file existence

# Define the function to extract website features
def extract_website_features(url):
    features = {}
    
    # Fetch website content
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content = response.text
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return None
    
    # URL and domain properties
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    ext = tldextract.extract(url)
    
    features['URLLength'] = len(url)
    features['DomainLength'] = len(domain)
    features['IsDomainIP'] = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else 0
    features['TLDLength'] = len(ext.suffix)
    features['NoOfSubDomain'] = len(ext.subdomain.split('.')) if ext.subdomain else 0
    features['IsHTTPS'] = 1 if parsed_url.scheme == 'https' else 0
    
    # HTML properties
    soup = BeautifulSoup(content, 'html.parser')
    features['LineOfCode'] = len(content.splitlines())
    features['HasTitle'] = 1 if soup.title and soup.title.string else 0
    features['HasDescription'] = 1 if soup.find('meta', {'name': 'description'}) else 0
    features['HasFavicon'] = 1 if soup.find('link', rel='icon') else 0
    features['NoOfPopup'] = len(soup.find_all('popup'))  # This might need adjustment; not commonly used in HTML
    features['NoOfiFrame'] = len(soup.find_all('iframe'))
    features['HasSubmitButton'] = 1 if soup.find('button', {'type': 'submit'}) else 0
    features['HasPasswordField'] = 1 if soup.find('input', {'type': 'password'}) else 0

    # Count specific characters and special characters in the URL
    features['NoOfLettersInURL'] = sum(c.isalpha() for c in url)
    features['NoOfDigitsInURL'] = sum(c.isdigit() for c in url)
    features['NoOfEqualsInURL'] = url.count('=')
    features['NoOfQMarkInURL'] = url.count('?')
    features['NoOfAmpersandInURL'] = url.count('&')
    
    special_chars = re.findall(r'[^a-zA-Z0-9]', url)
    features['NoOfOtherSpecialCharsInURL'] = len(special_chars) - features['NoOfEqualsInURL'] - features['NoOfQMarkInURL'] - features['NoOfAmpersandInURL']

    # External, empty, and self-referencing links
    links = [a['href'] for a in soup.find_all('a', href=True)]
    features['NoOfExternalRef'] = sum(1 for link in links if urlparse(link).netloc and urlparse(link).netloc != domain)
    features['NoOfEmptyRef'] = sum(1 for link in links if link == '#')
    features['NoOfSelfRef'] = sum(1 for link in links if urlparse(link).netloc == domain or link.startswith('/'))

    # Adding features calculations
    features['URLSimilarityIndex'] = calculate_url_similarity(url, domain)  # Simple similarity measure
    features['CharContinuationRate'] = calculate_char_continuation_rate(url)  # Based on URL characters
    features['TLDLegitimateProb'] = calculate_tld_legitimacy(ext.suffix)  # Check TLD against known legitimate TLDs
    features['URLCharProb'] = calculate_url_char_probability(url)  # Character distribution in URL
    features['HasObfuscation'] = 1 if any(char in url for char in ['%', '+', '=', '?']) else 0  # Basic obfuscation check
    features['NoOfObfuscatedChar'] = sum(1 for char in url if char in ['%', '+', '=', '?'])
    features['ObfuscationRatio'] = features['NoOfObfuscatedChar'] / features['URLLength'] if features['URLLength'] > 0 else 0
    features['LetterRatioInURL'] = features['NoOfLettersInURL'] / features['URLLength'] if features['URLLength'] > 0 else 0
    features['DigitRatioInURL'] = features['NoOfDigitsInURL'] / features['URLLength'] if features['URLLength'] > 0 else 0
    features['SpatialCharRatioInURL'] = features['NoOfOtherSpecialCharsInURL'] / features['URLLength'] if features['URLLength'] > 0 else 0
    features['DomainTitleMatchScore'] = calculate_domain_title_match_score(soup.title.string if soup.title else "", ext.domain)  # Match score based on title and domain
    features['URLTitleMatchScore'] = calculate_url_title_match_score(url, soup.title.string if soup.title else "")  # URL to Title match score
    features['Robots'] = 1 if soup.find('meta', attrs={'name': 'robots'}) else 0  # Check for robots meta tag
    features['IsResponsive'] = 1 if soup.find(attrs={'name': 'viewport'}) else 0  # Check for responsive design
    features['NoOfURLRedirect'] = 0  # Placeholder; actual check requires further implementation
    features['NoOfSelfRedirect'] = 0  # Placeholder; actual check requires further implementation
    features['HasSocialNet'] = 1 if soup.find('a', href=re.compile(r'social', re.IGNORECASE)) else 0  # Check for social media links
    features['HasHiddenFields'] = len(soup.find_all('input', type='hidden'))  # Count hidden fields
    features['Bank'] = 1 if re.search(r'bank|finance|loan|credit', url, re.IGNORECASE) else 0  # Check for banking-related terms
    features['Pay'] = 1 if re.search(r'pay|payment|checkout|purchase', url, re.IGNORECASE) else 0  # Check for payment-related terms
    features['Crypto'] = 1 if re.search(r'crypto|bitcoin|ethereum', url, re.IGNORECASE) else 0  # Check for cryptocurrency-related terms
    features['HasCopyrightInfo'] = 1 if soup.find('footer', text=re.compile(r'copyright', re.IGNORECASE)) else 0  # Check for copyright info in footer
    features['NoOfImage'] = len(soup.find_all('img'))  # Count of images
    features['NoOfCSS'] = len(soup.find_all('link', rel='stylesheet'))  # Count of CSS files
    features['NoOfJS'] = len(soup.find_all('script'))  # Count of JS files
    features['HTTPS'] = features['IsHTTPS']  # Copy HTTPS flag

    return features

# Helper functions for various calculations
def calculate_url_similarity(url, domain):
    """A simple measure of similarity between the URL and its domain."""
    return sum(1 for char in url if char in domain) / len(domain) if domain else 0

def calculate_char_continuation_rate(url):
    """Calculate the rate of continuation in characters (dummy implementation)."""
    return len(url) / len(set(url))  # Length of URL divided by number of unique characters

def calculate_tld_legitimacy(tld):
    """Check if the TLD is in a list of known legitimate TLDs (dummy implementation)."""
    legitimate_tlds = ['com', 'org', 'net', 'edu', 'gov', 'io', 'co', 'info']
    return 1 if tld in legitimate_tlds else 0

def calculate_url_char_probability(url):
    """Calculate the probability of character usage in the URL (dummy implementation)."""
    char_count = len(url)
    if char_count == 0:
        return 0
    return {char: url.count(char) / char_count for char in set(url)}  # Character distribution

def calculate_domain_title_match_score(title, domain):
    """Check if the domain is present in the title and score it."""
    return 1 if domain in title.lower() else 0

def calculate_url_title_match_score(url, title):
    """Calculate a score based on how much the URL relates to the title."""
    return sum(1 for word in title.split() if word.lower() in url.lower()) / len(title.split()) if title else 0

# Main function to create CSV with website features
def website_to_csv(url, output_file='PhiUSIIL_Phishing_URL_DatasetCustom.csv'):
    features = extract_website_features(url)
    if features:
        # Check if the file already exists to determine whether to write the header
        file_exists = os.path.isfile(output_file)

        # Create a DataFrame with a single row and export to CSV, appending if file exists
        df = pd.DataFrame([features])  # Wrap features in a list to create a DataFrame
        df.to_csv(output_file, mode='a', index=False, header=not file_exists)  # Append if file exists
        print(f"Features saved to {output_file}")

# User input for a single website
website_url = input("Enter a website URL to analyze: ")

# Run the function to save data to CSV
website_to_csv(website_url)
