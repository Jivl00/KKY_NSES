# KKY/NSES – Neuronové sítě a evoluční strategie

Tento repozitář obsahuje řešení semestrální práce z předmětu **KKY/NSES** na **Katedře kybernetiky**, **Fakulty aplikovaných věd**, **Západočeské univerzity v Plzni**.

Cílem projektu je implementovat dopředné neuronové sítě (jednovrstvou a dvouvrstvou) pro klasifikaci objektů do pěti tříd na základě dvou vstupních příznaků. Implementace probíhá **bez použití knihoven pro strojové učení**, pouze s využitím základních nástrojů jazyka Python.

---

## Obsah

* [Popis projektu](#popis-projektu)
* [Požadavky](#požadavky)
* [Použití](#použití)

---

## Popis projektu

Projekt se skládá ze dvou částí:

* **Jednovrstvá neuronová síť**
* **Dvouvrstvá neuronová síť**

Každá síť je samostatně trénována na dvou sadách trénovacích dat:

* `tren_data1.txt`
* `tren_data2.txt`

Formát trénovacích dat:

```
x1 x2 třída
```

Příklad:

```
6.9161 -5.7103 1
7.0601 -5.6065 1
7.2404 -5.3848 2
-7.0375 -5.4146 3
```

### Výstupy projektu:

* Průběh chyby v závislosti na trénovacích cyklech
* Rozhodovací oblast sítě ve vstupním prostoru

---

## Požadavky

* Python 3.8 nebo vyšší
* `numpy`
* `matplotlib`
* `seaborn`
* `IPython.display`
* `jupyter` (pro spuštění `showcase.ipynb`)

Instalaci požadavků lze provést příkazem:

```bash
pip install -r requirements.txt
```

---

## Použití

Pro spuštění vizualizací a trénování neuronových sítí použijte Jupyter notebook:

```bash
jupyter notebook SP/showcase.ipynb
```

Notebook obsahuje:

* Trénování obou typů sítí na obou datasetech
* Výpis a grafické znázornění průběhu chyby
* Vizualizaci rozhodovacích oblastí
