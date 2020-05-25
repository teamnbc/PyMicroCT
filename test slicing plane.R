# MicroCT stack is a s x s x s cube (s = 512 pixels).
# Pixels values in stack are defined using right hand reference frame (u,v,w).
# Ideally :   u along naso-occipital axix pointing towards the head.
#             v along medio-lateral axis pointing toward the left.
#             w along dorso-ventral axis pointing toward the dorsal side.
# Slicing plane defined by its center c(xc,yc,zc) and a vector c(a,b,c) of norm = 1.
# Slicing plane is square, with fixed width p (in pixels).
# Slicing plane is oriented so that two sides are parallel to sagittal plane.

if(0) {
  require(rgl)
  u=c(1,0,0)
  v=c(0,1,0)
  w=c(0,0,1)
  s=512
  p=201
  ori=c(227,248,227)
  vec=norml(c(0.5,0.5,-0.5))
  up=norml(as.numeric(t(vprod(v,vec))))
  vp=norml(as.numeric(t(vprod(vec,up))))
  clear3d()
  arrow3d(c(0,0,0),u*200,type='rotation',n=50,col=2)
  arrow3d(c(0,0,0),v*200,type='rotation',n=50,col=3)
  arrow3d(c(0,0,0),w*200,type='rotation',n=50,col=4)
  arrow3d(ori,ori+vec*150,type='rotation',n=50,col=4)
  arrow3d(ori,ori+up*150,type='rotation',n=50,col=2)
  arrow3d(ori,ori+vp*150,type='rotation',n=50,col=3)
  
  quads3d(512*t(matrix(c(0,0,0,1,0,0,1,0,1,0,0,1),nrow=3)),col=1,alpha=0.2)
  quads3d(512*t(matrix(c(0,0,0,1,0,0,1,1,0,0,1,0),nrow=3)),col=1,alpha=0.2)
  quads3d(512*t(matrix(c(0,1,0,1,1,0,1,1,1,0,1,1),nrow=3)),col=1,alpha=0.2)
  quads3d(512*t(matrix(c(0,0,1,1,0,1,1,1,1,0,1,1),nrow=3)),col=1,alpha=0.2)
  
  pCoords=do.call(rbind,lapply(seq(-100,100),function(i) {
    t(sapply(seq(-100,100),function(j) ori+i*up+j*vp))
  })); points3d(pCoords,col=1,pch=21,cex=5)

}



# Calculate the norm of each row of a 3-column matrix
norm <- function(x) {
  apply(x,1,function(y) sqrt(sum(y^2)))
}

# Normalize a vector
norml <- function(x) {
  x/sqrt(sum(x^2))
}

# Normalize each row of a 3-column matrix
Norml <- function(x) {
  x/sqrt(rowSums(x^2))
}

# 3D (cross and dot products) ####

# Scalar product of 2 vectors
sprod <- function(u,v) {
  sum(u*v)
}

# V = matrix (3 columns)
Sprod <- function(u,V) {
  colSums(u*t(V))
}

# U, V = matrix (3 columns)
SSprod <- function(U,V) {
  rowSums(U*V)
}

# Cross (vector) products
# vprod takes two vectors
vprod <- function(u,v) {
  rbind(u[2]*v[3]-u[3]*v[2],u[3]*v[1]-u[1]*v[3],u[1]*v[2]-u[2]*v[1]) # 3d cross (vector) product
}

# Vprod takes one vector and a matrix [1:n, 1:3]
Vprod <- function(u,V) {
  cbind(u[2]*V[,3]-u[3]*V[,2],u[3]*V[,1]-u[1]*V[,3],u[1]*V[,2]-u[2]*V[,1])
}

# VVprod takes two matrices [1:n, 1:3]
VVprod <- function(U,V) {
  cbind(U[,2]*V[,3]-U[,3]*V[,2],
        U[,3]*V[,1]-U[,1]*V[,3],
        U[,1]*V[,2]-U[,2]*V[,1])
}

Vproj <- function(u,V) {
  # u: vector defining projection plane (reference vector)
  # V: 3-column matrix containing vector coordinates
  un=norml(u) # normalize reference vector
  v=Norml(-Vprod(un,Norml(Vprod(un,Norml(V)))))
  s=SSprod(v,V)
  s*v # V projected onto plane
}