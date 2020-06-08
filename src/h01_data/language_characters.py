"""
    This file is used to define a character set of each language in order to
    clean the wiki data by detecting and removing sentences with words from a
    different character set.
    The non-ascii characters are defined using this site:
    https://r12a.github.io/app-charuse/
"""

import string

def get_latin_base():
    return string.ascii_lowercase

def get_character_set(language):
    character_sets = {
        'en': get_latin_base(),
        'fi': get_latin_base() + 'äåöäåöššžž',
        'yo': get_latin_base() + 'áàāéèēẹe̩ẹ́é̩ẹ̀è̩ẹ̄ē̩íìīóòōọo̩ọ́ó̩ọ̀ò̩ọ̄ō̩úùūṣs̩',
        'he': 'אבגדהוזחטיךכלםמןנסעףפץצקרשת',
        'id': get_latin_base(),
        'ta': 'ஃஅஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஶஷஸஹ' + \
              'ாிீுூெேைொோௌ்ௗ',
        'tr': get_latin_base() + 'âçöüâçöüğğiışş'
    }
    try:
        chars = character_sets[language]
    except KeyError:
        print('No character set defined for ', language)
    return chars
