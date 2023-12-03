# Rapport TP3 - Maxime Fromont, Bastien Pili, Elyas Tigre
## Ce rapport traitera des difficultés rencontrés, des solutions apportés, et des réponses obtenues

Comme conseillé dans le sujet, nous avons décidé d'utiliser la librairie OpenCV. De plus, pour des raisons de simplicité et de praticité, nous développerons nos expérimentations et les réponses aux questions du TP en Python.

### Exercice 1

Nous avons d'abord eu du mal à différencier le résultat des différents types de détecteur. C'est en analisant la documentation d'OpenCV que nous nous sommes rendu compte de l'importance de l'ajustement des paramètres. Ayant testé chaque détecteur un par un, nous avons finalement décidé de choisir le détecteur "ORB (Oriented FAST and Rotated BRIEF)" dont la documentation est détaillé ici : https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

En effet, le détecteur ORB est rapide et efficace dans la détection des points d'intérêt, ce qui est essentiel étant donné le nombre de fragments d'images, il est capable de détecter des points d'intérêt robustes même en présence de rotations, ce qui pourrait potentiellement être très utile par la suite et il offre un bon compromis entre la vitesse de calcul et la qualité des descripteurs générés.

Nous avons aussi condidéré utiliser FAST (donc la documentation est détaillé ici : https://docs.opencv.org/4.x/df/d0c/tutorial_py_fast.html) mais nos tests démontre une meilleur efficacité d'ORB, qui à d'autant plus été conseillé par M. ALDEA.

Afin d'acompagner le détecteur ORB, il semblerait de part les conseils du web et en particulier de StackOverFlow que le descripteur idéal soit BRIEF (Binary Robust Independent Elementary Features). La documentation de BRIEF est disponible juste ici : https://docs.opencv.org/4.x/dc/d7d/tutorial_py_brief.html.

Comme ORB, BRIEF est conçu pour être rapide à calculer et avoir une utilisation de mémoire satisfesante, ce qui est très important encore une fois étant données le nombre de fragments. De plus BRIEF est souvent utilisé conjointement avec ORB, et leur combinaison peut offrir de bons résultats en termes d'association de points d'intérêt selon l'expérience des développeurs experimenté sur le sujet.

Pour conclure cette exercice, vous retrouverez notre expérimentation finale dans le fichier TP3-EX1_Fromont-Pili-Tigre.py.

Cette expérimentation finale, bien qu'elle fonctionne, n'arrive pas à trouver de keypoints suffisants pour un peu plus d'une centaine de fragment. Le nombre de keypoints sufisant ici, est de 10. Si le nombre de keypoint est sufisant, nous imprimons alors les matches et leurs détails dans associations.txt. Voici un exemple de sortie :

Matches for frag_eroded_0.png:
Distance: 9.0, TrainIdx: 420, QueryIdx: 495
Distance: 11.0, TrainIdx: 94, QueryIdx: 111
Distance: 12.0, TrainIdx: 113, QueryIdx: 20
Distance: 12.0, TrainIdx: 73, QueryIdx: 120
Distance: 13.0, TrainIdx: 277, QueryIdx: 345
Distance: 13.0, TrainIdx: 566, QueryIdx: 599
Distance: 14.0, TrainIdx: 238, QueryIdx: 297
Distance: 14.0, TrainIdx: 716, QueryIdx: 756
Distance: 16.0, TrainIdx: 428, QueryIdx: 441
Distance: 16.0, TrainIdx: 422, QueryIdx: 457

Nous avons essayé d'utiliser plusieurs autres méthodes, comme en essayant d'utiliser un détecteur SIFT, ou plusieurs détecteurs en même temps maos nous n'avons malheuresement pas réussie à obtenir plus de 10 keypoints pour chaque fragments.

### Exercice 2

#### Question 1

K représente le nombre minimum d'associations nécessaire pour calculer une paramétrisation x, y, θ pour la transformation appliquée au fragment. En général, pour estimer une transformation géométrique entre deux ensembles de points, K doit être égal au nombre minimum de correspondances requis pour obtenir un modèle de transformation fiable. Souvent, K est déterminé par le type de transformation utilisé et la méthode de calcul des paramètres de transformation.

La paramétrisation x, y, θ peut être calculée comme suit à partir des coordonnées des associations :

    x et y représentent les composantes de translation (décalage) dans les directions horizontale et verticale respectivement. Ces valeurs sont généralement obtenues en moyennant les différences entre les positions des points clés correspondants dans les deux images.

    θ représente l'angle de rotation. Il est généralement calculé en tenant compte des différences d'orientation entre les keypoints correspondants.

En utilisant les associations (correspondances) entre les points clés de l'image principale et ceux du fragment, on peut appliquer des méthodes telles que la méthode des moindres carrés ou RANSAC pour estimer ces paramètres (x, y, θ) de la transformation géométrique qui aligne au mieux le fragment avec l'image principale.

#### Question 2

Vous trouverez cette implémentation dans le fichier TP3-EX2-Q2_Fromont-Pili-Tigre.py. Voici un exemple de sortie :

Nombre de correspondances dans le sous-ensemble cohérent : 8
Index des points correspondants dans le sous-ensemble cohérent : [271, 211, 483, 155, 884, 64, 918, 718]
Paramètres de la transformation estimée (x, y, θ) : [[-2.29041063e-02 -6.14444154e-02  1.88806370e+02]
 [ 6.14444154e-02 -2.29041063e-02 -4.69055217e+01]]

 #### Question 3

 Vous trouverez l'implémentation nécéssaire pour répondre à cette question dans le fichier TP3-EX2-Q3_Fromont-Pili-Tigre. Voici un exemple de sortie :

 Matrice de transformation optimale :
[[-1.70682369e-03 -4.81246437e-03  1.40262069e+02]
 [ 4.81246437e-03 -1.70682369e-03  7.17968083e+01]]

 ### Exercice 3

 Vous trouvez le code récréant l'image final à partir des fragments dans le fichier TP3-EX3_Fromont-Pili-Tigre.
 Pour l'instant le résultat n'est pas du tout concluant.

 ### Exercice 4
Nous proposons la stratégie suivante, en se basant sur la préservation des distances entre les pixels pour filtrer les associations :
 - On calcul les distances entre les points d'intérêt dans les deux images après les transformations candidates.
 - On compare les distances calculées pour chaque paire de points correspondants.
 - On filtre les associations qui ne conservent pas les distances de manière cohérente dans les images transformées.

 Cela permettrait non seulement d'être très efficace en raison de l'utilisation d'une propriété de mathématique directe mais aussi d'être moins intensif en calculs que RANSAC. Ceci dit, cela aurait aussi pour inconvénient d'être très sensible aux erreurs de correspondance dues au bruit ou à des transformations non linéaires. De plus, cette stratégie ne serait applicable qu'aux transformations qui préservent les distances entre les pixels.

L'utilisation de cette stratégie pourrait être pertinente dans le cas ou nous avons une connaissance préalable sur le type de transformation à appliquer (par exemple, uniquement les transformations affines). Toutefois, si les transformations peuvent être complexes ou si le bruit est élevé, cette méthode pourrait ne pas être aussi robuste que RANSAC, qui peut mieux gérer les incohérences et les transformations non linéaires.