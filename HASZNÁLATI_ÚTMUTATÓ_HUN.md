# Antioxidáns analízis – Voltammetria

Ez a szkript a voltammetriás adatokat a standard addíciós módszerrel értékeli ki.

## Követelmények
- Python 3
- numpy
- pandas
- matplotlib
- scipy

Telepítés:
pip install numpy pandas matplotlib scipy

## Használat

1. Helyezze a `.txt` mérési fájlokat egy mappába
2. Futtassa a szkriptet:
python antioxidáns.py

3. Írja be:
- mappa elérési útja
- aszkorbinsav koncentráció

## Kimenet

A szkript a következőket generálja:
- Voltammogramok (PDF)
- Kalibrációs görbe (PDF)
- Csúcsáram táblázat (PDF)

## Módszer

A hozzáadott koncentráció kiszámítása a következőképpen történik:

c_added = (c_AA * V_added) / (V_sample + V_added)

Az ismeretlen koncentrációt a kalibrációs görbe x metszéspontjából határozzuk meg.
