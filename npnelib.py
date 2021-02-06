import tensorflow as tf
import numpy as np

@tf.function
def NPq(r,fk,Ib,Itw=None):
    """ implements NPq (non-parametric committor optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """
    
    if Itw is None : Itw=tf.ones_like(r)
    fk=fk*(1-Ib)

    dfk=tf.roll(fk,-1,1)-fk
    akj=tf.tensordot(dfk*Itw,dfk,axes=[1,1])
    
    dr=-(tf.roll(r,-1,0)-r)
    b=tf.tensordot(dfk,dr*Itw, 1)
    b=tf.reshape(b, [b.shape[0],1])

    al_j=tf.linalg.lstsq(akj,b,fast=False)
    al_j=tf.reshape(al_j, [al_j.shape[0]])
    
    rn=r+tf.tensordot(al_j,fk,1)
    rn=tf.clip_by_value(rn,0,1)
    return rn 

@tf.function
def NPNEq(r,fk,Ib,It):
    """ implements NPNEq (non-parametric non-equilbrium committor
    optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    rmin,rmax - minimal and maximal to clip the updated RC
    """
    fk=fk*(1-Ib)
    
    dfj=fk-tf.roll(fk,-1,1)
    akj=tf.tensordot(fk*It,dfj,axes=[1,1])

    dr=tf.roll(r,-1,0)-r
    b=tf.tensordot(fk,dr*It, 1)
    b=tf.reshape(b, [b.shape[0],1])

    al_j=tf.linalg.lstsq(akj,b,fast=False)
    al_j=tf.reshape(al_j, [al_j.shape[0]])
    
    rn=r+tf.tensordot(al_j,fk,1)
    rn=tf.clip_by_value(rn,0,1)
    return rn 

@tf.function
def NPNEw(r,fk,It):
    """ implements NPNEw (non-parametric non-equilbrium re-weighting factors 
    optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """
    
    dfk=fk-tf.roll(fk,-1,1)

    b=-tf.tensordot(dfk,r*It, 1)
    b=tf.reshape(b, [b.shape[0],1])
    scale=tf.math.reduce_sum(1-r*It)
    scale=tf.reshape(scale,[1,1])
    b=tf.concat((b,scale),0)
    
    ones=tf.reshape(It,[1,It.shape[0]])
    dfk=tf.concat((dfk*It,ones),0)
    akj=tf.tensordot(dfk,fk,axes=[1,1])
    
    al_j=tf.linalg.lstsq(akj,b,fast=False)
    al_j=tf.reshape(al_j, [al_j.shape[0]])
    
    rn=r+tf.tensordot(al_j,fk,1)
    
    return rn


@tf.function
def basis_poly_ry(r,y,n,fenv=None):
    """computes basis functions as terms of polynomial of variables r and y

    r is the putative RC time-series
    y is a randomly chosen collective variable or coordinate to improve r
    n is the degree of the polynomial
    fenv is common envelope to focus optimization on a particular region
    """
    r=r/tf.math.reduce_max(tf.math.abs(r))
    y=y/tf.math.reduce_max(tf.math.abs(y))

    if fenv is None:
        f=tf.ones_like(r)
    else:
        f=tf.identity(fenv)
        
    fk=[]
    for ir in range(n+1):
        fy=tf.identity(f)
        for iy in range(n+1-ir):
            fk.append(fy)
            fy=fy*y
        f=f*r
    return tf.stack(fk)

@tf.function
def basis_poly_r(r,n,fenv=None):
    """computes basis functions as terms of polynomial of variable r

    r is the putative RC time-series
    y is a randomly chosen collective variable or coordinate to improve r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    n is the degree of the polynomial
    fenv is common envelope to focus optimization on a particular region
    """
    r=r/tf.math.reduce_max(tf.math.abs(r))

    if fenv is None:
        f=tf.ones_like(r)
    else:
        f=tf.identity(fenv)
    
    fk=[]
    for ir in range(n+1):
        fk.append(f)
        f=f*r
    return tf.stack(fk)

def comp_Zh(lx,dx,lw=[]):
    """ compute Zh, histogram-based partition function/probability
    lx - list of coordinates, e.g., a trajectory time-series
    dx - bin size
    lw - list of weights, for re-weighting.
    """
    import math
    zh={}
    if len(lw)>0:
        for x,w in zip(lx,lw):    
            x=math.floor(x/dx+0.5)*dx
            zh[x]=zh.get(x,0)+w
    else:
        for x in lx:    
            x=math.floor(x/dx+0.5)*dx
            zh[x]=zh.get(x,0)+1
        
    lx=list(zh.keys())
    lx.sort()
    ly=[zh[x]/dx for x in lx]
    return lx,ly


def comp_Zq(lx, itraj, dt=1, strict=False, dx=1e-3):
    """computes non-equilibrium committor validation criterion Z_q,
 
    lx - list of coordinates, a trajectory time-series
    itraj - trajectory index time-series:
        tells to which short trajectory point X(i) belongs to
    dt - the value of \Delta t
    strict - whether to construct the exact profile, accurate representation of step functions
    dx - bin size for coarse-graining
    """
    return comp_Zca(lx, a=1, itraj=itraj, dt=dt, strict=strict, dx=dx, eq=False)



def comp_Zca(lx, a, itraj=[], lw=[], dt=1, strict=False, dx=1e-3, mindx=1e-3, eq=True):
    """computes cut-based free energy profiles Z_C,\alpha for non-equilbrium and equilbrium cases
    non-equilbrium case (eq=False) computes only outgoing part of the Z_C,\alpha,
        which for a=1 (\alpha=1) gives Z_q - the non-equilbrium committor criterion
    equilbrium case (eq=True), computes outgoing and ingoing parts of Z_C,\alpha
    can be used with a single trajectory and with ensamble of trajectories
    accepts re-weighting factors
    
    lx - list of coordinates, a trajectory time-series
    a  - the value of alpha
    itraj - trajectory index time-series:
        tells to which short trajectory point X(i) belongs to
        if itraj=[], a single long trajectory is assumed
    lw - list of weights or re-weighting factors
    dt - the value of \Delta t
    strict - whether to construct the exact profile, accurate representation of step functions
    dx - bin size for coarse-graining
    mindx - minimal value of dx:
        used to suppress errors for a<0
    eq - flag to compute equilibrium profile (eq=True), by including in-going transitions
    """
    import math
    dzc={}
    tmax=len(lx)
    for i in range(0,tmax-dt):
        if len(itraj)>0 and itraj[i]!=itraj[i+dt]:continue
        x=lx[i+dt]
        startx=lx[i]
        if dx!=None and not strict: 
            x=math.floor(x/dx+0.5)*dx
            startx=math.floor(startx/dx+0.5)*dx
        d=abs(x-startx)
        if a<0 and d<mindx:d=mindx
        if a==0: d=1 
        else: d=float(d)**a
        if len(lw)>0:d=d*lw[i]    
        if startx<x:
            dzc[startx]=dzc.get(startx,0)+d
            if eq:dzc[x]=dzc.get(x,0)-d
        else:  
            dzc[startx]=dzc.get(startx,0)-d
            if eq:dzc[x]=dzc.get(x,0)+d
    
    keys=list(dzc.keys())
    keys.sort()
    lx=[]
    ly=[]
    z=0
    scale=1/dt
    if eq:scale=1/dt/2
    for x in keys:
        lx.append(x)
        ly.append(float(z))
        z=z+dzc[x]*scale
        if strict:
            lx.append(x)
            ly.append(float(z))
    if not strict and len(ly)>1: # ly[0] equals 0, but a non-zero value is more convenieint for further analysis
        del ly[0], lx[0] 
    return lx,ly 

def comp_ekn_tp(xtraj,x0,x1,itraj=[],dx=1e-3,dt=1):
    """ transition path segment counting scheme
    computes an equilibrium kinetic network - the number of transition between different points
    it should be followed by function comp_Zca_ekn, which computes the cut-profiles
    
    xtraj - trajectory time-series
    x0 - the position of the boundary of the left boundary state
    x1 - the position of the boundary of the right boundary state
    itraj - trajectory index time-series:
        tells to which short trajectory point X(i) belongs to
        if itraj=[], a single long trajectory is assumed
    dx - bin size for coarse-graining
    dt - the value of \Delta t
    """
    import math
    def processsegm(traj, x0,x1): # process a TP segment
        n=len(traj)
        if n<2:return
        firstb=traj[0]<=x0 or traj[0]>=x1
        lastb=traj[-1]<=x0 or traj[-1]>=x1
        for i in range(1,n): # from i-dt to i
            j=i-dt
            if j<0:
                if not firstb:continue
                j=0
            key=traj[j],traj[i]
            ekn[key]=ekn.get(key,0)+1
        if lastb:
            for i in range(max(n-dt,1),n-1):
                key=traj[i],traj[-1]
                ekn[key]=ekn.get(key,0)+1
        if firstb and lastb and dt>n-1:
            key=traj[0],traj[-1]
            ekn[key]=ekn.get(key,0)+dt-n+1
    
    def processtraj(traj):
        lx=[]
        for x in traj:
            if dx!=None:x=math.floor(x/dx+0.5)*dx
            if x<=x0:
                lx.append(x0)
                processsegm(lx,x0,x1)
                lx=[x0]
                continue
            if x>=x1:  
                lx.append(x1)
                processsegm(lx,x0,x1)
                lx=[x1]
                continue
            lx.append(x)
        processsegm(lx,x0,x1)

    ekn={}
    if dx!=None: 
        x0=math.floor(x0/dx+0.5)*dx
        x1=math.floor(x1/dx+0.5)*dx

    lx=[]
    if len(itraj)>0: # iterate over an ensemble of trajectories
        ct=itraj[0]
        for x,it in zip(xtraj,itraj):
            if ct==it:
                lx.append(x)
            else: 
                processtraj(lx)
                ct=it
                lx=[x]
        processtraj(lx)
    else:
        processtraj(xtraj)
        
    for ij in ekn:ekn[ij]=float(ekn[ij])/dt   
    return ekn

def comp_Zca_ekn(ekn,a,dx=None,strict=False,eq=True):
    """computes cut-based free energy profiles Z_C,\alpha using the transition
    pathway segment summation scheme, for non-equilbrium and equilbrium cases
    used together with comp_ekn_tp function
    
    ekn - an equilibrium kinetic network computed by comp_ekn_tp
    a  - the value of alpha
    dx - bin size for coarse-graining
    strict - whether to construct the exact profile, accurate representation of step functions
    eq - flag to compute equilbrium profile, by including in-going transitions
    """
    import math
    dzc={}
    for y,x in ekn:
        d=abs(y-x)**a*ekn[(y,x)]
        if dx!=None: 
            x=math.floor(x/dx+0.5)*dx
            y=math.floor(y/dx+0.5)*dx
        if y<x:
            dzc[y]=dzc.get(y,0)+d
            if eq:dzc[x]=dzc.get(x,0)-d
        else:  
            dzc[y]=dzc.get(y,0)-d
            if eq:dzc[x]=dzc.get(x,0)+d
            
    keys=list(dzc.keys())
    keys.sort()
    lx=[]
    ly=[]
    z=0
    scale=1
    if eq:scale=0.5
    for x in keys:
        lx.append(x)
        ly.append(float(z))
        z=z+dzc[x]*scale
        if strict:
            lx.append(x)
            ly.append(float(z))
    if not strict and len(ly)>1: # ly[0] equals 0, but a non-zero value is convenieint for further analysis
        del ly[0],lx[0] 
    return lx,ly 



def tonatural(lq,dx,dtsim=1,itraj=[],lw=[],fixZhA=True,zcm1=False):
    """ transforms putative commitor coordinate to the natural coordinate,
    where diffusion coefficient is constant. The following equations are used
    Zc1=dt*D*Zh 
    D=Zc1/(Zh*dt)
    dy/dx=D^-1/2
    
    lq - committor time-series
    dx - bin size to compute the profiles and integrate
    dtsim - the trajectory/simulation sampling interval in time units in which D=1.
    itraj - trajectory index time-series:
        tells to which short trajectory point X(i) belongs to
        if itraj=[], a single long trajectory is assumed  
    lw - list of weights, for re-weighting,
        if lw=[], no re-weighting
    fixZhA - the leftmost value of Z_H, Z_H for state A, usually has a very large value,
        which leads to unphysical results. fixZhA=True assigns it to the next value 
    zcm1 - whether to use Z_C,-1 to estimate Z_H. Z_C,-1 gives a more robust estimate
        of Z_H if the sampling interval is sufficiently small.
    """
    import math

    if zcm1:
        lx,lzcm1=comp_Zca(lq,a=-1,itraj=itraj,dx=dx,dt=1,lw=lw,eq=True)
        lzh=[2*x for x in lzcm1]
    else:
        lx,lzh=comp_Zh(lq,lw=lw,dx=dx)
    lx1,lzc1=comp_Zca(lq,a=1,itraj=itraj,dx=dx,dt=1,lw=lw,eq=True)
    
    if fixZhA:lzh[0]=lzh[1]
    y=0
    x2y={}
    dydx={}
    xl=0
    x2y[xl]=0
    for x,zh,zc1 in zip(lx1,lzh,lzc1):
        dydx[xl]=(zh*dtsim/zc1)**0.5
        y=y+dydx[xl]*(x-xl)
        x2y[x]=y
        xl=x
    dydx[1.0]=0
    x2y[1.0]=y
    ly=[]
    for q in lq:
        q0=math.floor(q/dx+0.5)*dx
        try: qn=x2y[q0]+(q-q0)*dydx[q0]
        except: pass
        ly.append(qn)
    return ly