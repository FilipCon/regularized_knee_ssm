function [fit_shape_nii] = mesh2label_map(shape, ref_labelmap, n)

VolumeTarget = ref_labelmap.img;

b = ref_labelmap.hdr.hist.quatern_b;
c = ref_labelmap.hdr.hist.quatern_c;
d = ref_labelmap.hdr.hist.quatern_d;
a = sqrt( 1 - b^2 - c^2 - d^2);
quat = [ a b c d ];
rotm = quat2rotm(quat);

q_offset = [ref_labelmap.hdr.hist.qoffset_x;...
    ref_labelmap.hdr.hist.qoffset_y;
    ref_labelmap.hdr.hist.qoffset_z];

xyz = shape.vertices' - q_offset;

q = ref_labelmap.hdr.dime.pixdim(1);
dim_x = ref_labelmap.hdr.dime.pixdim(2);
dim_y = ref_labelmap.hdr.dime.pixdim(3);
dim_z = ref_labelmap.hdr.dime.pixdim(4);

R = rotm*[dim_x 0 0; 0 dim_y 0; 0 0 q*dim_z];

temp = R\xyz;
shape.vertices = temp';

imgsize = [size(VolumeTarget)];


gridX = [1:imgsize(1)];
gridY = [1:imgsize(2)];
gridZ = [1:imgsize(3)];

% VolumeSSm = polygon2voxel(shape,imgsize,'none',false);
VolumeSSm = VOXELISE2(gridX,gridY,gridZ,shape);



VolumeSSm = uint8(imfill(VolumeSSm,'holes'));
VolumeSSm(VolumeSSm==1)=n;

fit_shape_nii = ref_labelmap;
fit_shape_nii.img = VolumeSSm;
