# superscreen-squids

SuperScreen models for scanning SQUID sensors

## `squids.show_all`

Usage:

```python
python -m squids.show_all
```

![squids.show_all](images/show_all.png)

```python
python -m squids.show_all --draw
```

![squids.show_all --draw](images/show_all--draw.png)

```python
python -m squids.show_all --draw --same-scale
```

![squids.show_all --draw --same-scale](images/show_all--draw--same-scale.png)

## `squids.mutuals``

Usage:

```python
python -m squids.mutuals
```

Output:

```
squids.hypres.small
-------------------
285.50878980023543 magnetic_flux_quantum / ampere
0.590384739582329 picohenry
-------------------------------------------------

squids.ibm.small
----------------
72.8267388114595 magnetic_flux_quantum / ampere
0.15059359558743207 picohenry
-----------------------------------------------

squids.ibm.medium
-----------------
158.75456595635796 magnetic_flux_quantum / ampere
0.3282780650824389 picohenry
-------------------------------------------------

squids.ibm.large
----------------
601.280775544196 magnetic_flux_quantum / ampere
1.2433487400997283 picohenry
-----------------------------------------------

squids.ibm.xlarge
-----------------
1576.6470894734962 magnetic_flux_quantum / ampere
3.2602442186922795 picohenry
-------------------------------------------------

squids.huber
------------
882.0365495331502 magnetic_flux_quantum / ampere
1.823905032705215 picohenry
------------------------------------------------
```