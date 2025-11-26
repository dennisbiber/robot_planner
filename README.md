# A tool to simulate a robot driving on a map using matplotlib
W.I.P.

# To Build lane graph
```
- python makeLanes.py ./config/map1.yml
```
map1.yml is an example map

# To Run map
```
- python simControl.py mapTest.yml <starting lane ID> <ending lane ID>
example
- python simControl.py mapTest.yml E1 E2
```

# Building a map
```
to buld a map, every enter/exit edge of a lane must equal an edge of an intersection

it currently has some limitations in the allowed geometries of map lanes. 
```

![Map of 'mapTest.yml" built off example config 'map1.yml' with route from E1 to E2](assets/Figure_1.png)