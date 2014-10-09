MLC-BMaD
========

This project contains a multi-label classifier based on Boolean
matrix decomposition. Boolean matrix decomposition is used to extract,
from the full label matrix, latent labels representing useful Boolean
combinations of the original labels. Base level models predict latent
labels, which are subsequently transformed into the actual labels by
Boolean matrix multiplication with the second matrix from the
decomposition. 

Build
=====


We use maven as a build tool, so just run

```
mvn clean install
```

to build MLC-BMaD. 

You can include BMaD using our maven repository:

```
<repository>
	<id>jgu</id>
	<name>jgu</name>
	<url>http://fantomas.informatik.uni-mainz.de/mvnrepo/repository</url>
</repository>
```
```
<dependency>
	<groupId>org.kramerlab</groupId>
	<artifactId>mlcbmad</artifactId>
	<version>1.0</version>
</dependency>
```


Citation
========

If you want to cite MLC-BMaD in your publication, please cite the
following ACM SAC paper:

```
Jörg Wicker, Bernhard Pfahringer, and Stefan Kramer.
 “Multi-label Classification Using Boolean Matrix Decomposition”.
Proceedings of the 27th Annual ACM Symposium on Applied Computing.
ACM New York, NY, USA, 2012, pp. 179-186.
```

Bibtex entry:

```
@inproceedings{wicker2012multi,
keywords={selected,conference, Boolean matrix decomposition, associations, multi-label classification},
 author = {Wicker, J\"{o}rg and Pfahringer, Bernhard and Kramer, Stefan},
 title = {Multi-label Classification Using {Boolean} Matrix Decomposition},
 booktitle = {Proceedings of the 27th Annual {ACM} Symposium on Applied Computing},
 series = {SAC '12},
 year = {2012},
 isbn = {978-1-4503-0857-1},
 location = {Trento, Italy},
 pages = {179--186},
 numpages = {8},
 url = {http://doi.acm.org/10.1145/2245276.2245311},
 doi = {10.1145/2245276.2245311},
 acmid = {2245311},
 publisher = {ACM},
 address = {New York, NY, USA}
}
```