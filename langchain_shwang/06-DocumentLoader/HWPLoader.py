from typing import Any, Dict, List, Optional, Iterator
import olefile
import zlib
import struct
import re
import unicodedata
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
