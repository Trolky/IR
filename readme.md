# Wowhead News Crawler

Crawler pro automatizované stahování článků ze sekce **News** na webu [Wowhead](https://www.wowhead.com) pomocí **Selenium** a ukládání výsledků do formátu **JSON Lines (.jsonl)**.

Projekt využívá:
- Firefox WebDriver (headless režim)
- Selenium pro zpracování dynamického webu
- Xpath

---

## Základní Funkce

- Prochází stránkování sekce **News**
- Filtruje pouze validní URL článků
- Deduplikuje opakující se články
- Stahuje:
  - Titulek
  - Autora
  - Datum publikace
  - Obsah článku
- Ukládá každý článek jako jeden JSON záznam (JSONL formát)
- Loguje průběh do souboru `wowhead_downloader.log`

---
