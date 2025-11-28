SetFactory("OpenCASCADE");
h = 0.01;
Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;
// Spline
Point(1) = {0,0.007,0,h};
Point(2) = {0,0.025,0.1,h};
Point(3) = {0,0.007,0.2,h};
Spline(1) = {1,2,3};
// Lines to for end faces
Point(6) = {0,0,0.2,h};
Point(7) = {0,0,0,h};
Line(2) = {3,6};
Line(3) = {7,1};

// Center axis line
Line(4) = {6,7};

Curve Loop(3) = {1,2,3,4};
Plane Surface(3) = {3};

// Rotate
Extrude { {0,0,1}, {0,0,0}, 2*Pi} { Surface{3} ;}


// Center cross-section
Disk(7) = {0, 0, 0.1, 0.025};

BooleanFragments{Volume{1}; Delete;}{Surface{7}; Delete; }


Physical Volume(1) = {1,2};
Mesh 1;
Physical Surface(1) = {9}; // bottom face
Physical Surface(2) = {11}; // top faces
Physical Surface(3) = {7}; // mid CSA face