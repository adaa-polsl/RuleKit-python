## Inicjalizacja

Przed użyciem bilbioteki trzeba ją zainicjalizować (uruchomić JVM):

```python
from rulekit.main import RuleKit

RuleKit.init()
```

możliwe jest także odczytanie wersji jar'ki RuleKit'a:

```python
>> RuleKit.version
>> 2.0.2
```

## Przekazywanie danych do modelu

Model oczkuje danych w formacie analogicznym do tego z scikit. Można przekazywać tablice numpy, zwykłe listy, lub dataframy pandasa. 

### Używanie numpy

```python

x = np.array([0, 0], [1, 0], [1, 1], [0, 1])
y = np.array([0, 0, 1, 0])

clf = rulekit.RuleClassifier()
clf.fit(x, y)

prediction = tree.predict(test_x)
```

### Używanie pandas

```python
train_x = data_frame[['Age', 'Gender', 'Payment Method']]
test_x = test_data_frame[['Age', 'Gender', 'Payment Method']]

tree = rulekit.RuleClassifier()
tree.fit(train_x, train_y)

prediction = tree.predict(test_x)
```

> Biblioteka dla nominalnych labeli zwraca tablice typu `str` a nie byte tak jak ma to miejsce w pandasie

np.
```python
test_y = data_frame['Future Customer'].to_numpy(dtype=str)

prediction = tree.predict(test_y)

accuracy = metrics.accuracy_score(y, prediction)
```

## Operatory

Dostępne są następujące operatory odpowiadające klasą z javy:
* `classification.RuleClassifier` - ClassificationSnC
* `classification.ExpertRuleClassifier` - ClassificationExpertSnC
* `regression.RuleRegressor` - RegressionSnC
* `regression.ExpertRuleRegressor` - RegressionExpertSnC
* `survival.SurvivalRules` - SurvivalLogRankSnC
* `survival.ExpertSurvivalRules` - SurvivalLogRankExpertSnC

## Konfigurowanie operatora

Wszystkie parametry operatora (z wyjątkiem reguł eksperckich) przekazywane są poprzez analogicznie nazwane parametry konstruktora.

## Nauczony model

Metoda `fit` zwraca obiekt `RuleSet`, można się też do niego dostać poprzez pole `model` operatora. Posiada on analogiczne pola do tych z bibliotki Java:
* `growing_time`: float
* `is_voting`: bool
* `pruning_time`: float
* `total_time`: float
* `parameters`: InductionParameters
* `rules`: List[Rule]
