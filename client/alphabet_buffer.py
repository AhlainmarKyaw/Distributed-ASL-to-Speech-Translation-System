from __future__ import annotations

class AlphabetBuffer:
    def __init__(self) -> None:
        self._text = ''

    @property
    def text(self) -> str:
        return self._text

    def add_letter(self, letter: str) -> None:
        if len(letter) == 1 and letter.isalpha():
            self._text += letter.upper()

    def space(self) -> None:
        if self._text and not self._text.endswith(' '):
            self._text += ' '

    def delete(self) -> None:
        self._text = self._text[:-1]

    def clear(self) -> None:
        self._text = ''

    def set_text(self, text: str) -> None:
        self._text = text
