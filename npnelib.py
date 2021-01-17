import tensorflow as tf
import numpy as np

@tf.function
def NPq(r,y,Ib,nr,ny,Itw=None,fenv=None,rmin=0,rmax=1):
    
    if Itw==None : Itw=tf.ones_like(Ib)
    nIb=1-Ib
    
    fk=[]
    if fenv==None:
        f=tf.identity(nIb)
    else:
        f=tf.identity(fenv*nIb)
    for ir in range(ny):
        fy=tf.identity(f)
        for iy in range(ny-ir):
            fk.append(fy)
            fy=fy*y
        f=f*r
    for ir in range(ny,nr):
        fk.append(f)
        f=f*r
    fk=tf.stack(fk)
    
    dfk=tf.roll(fk,-1,1)-fk
    akj=tf.tensordot(dfk*Itw,dfk,axes=[1,1])
    
    dr=-(tf.roll(r,-1,0)-r)
    b=tf.tensordot(dfk,dr*Itw, 1)
    b=tf.reshape(b, [b.shape[0],1])

    al_j=tf.linalg.lstsq(akj,b,fast=False)
    al_j=tf.reshape(al_j, [al_j.shape[0]])
    
    rn=r+tf.tensordot(al_j,fk,1)
    rn=tf.clip_by_value(rn,rmin,rmax)
    return rn 

@tf.function
def NPNEq(r,y,It,Ib,nr,ny,fenv=None,rmin=0,rmax=1):
    
    nIb=1-Ib

    fk=[]
    if fenv==None:
        f=tf.identity(nIb)
    else:
        f=tf.identity(fenv*nIb)
    for ir in range(ny):
        fy=tf.identity(f)
        for iy in range(ny-ir):
            fk.append(fy)
            fy=fy*y
        f=f*r
    for ir in range(ny,nr):
        fk.append(f)
        f=f*r
    fk=tf.stack(fk)
    
    dfj=fk-tf.roll(fk,-1,1)
    akj=tf.tensordot(fk*It,dfj,axes=[1,1])

    dr=tf.roll(r,-1,0)-r
    b=tf.tensordot(fk,dr*It, 1)
    b=tf.reshape(b, [b.shape[0],1])

    al_j=tf.linalg.lstsq(akj,b,fast=False)
    al_j=tf.reshape(al_j, [al_j.shape[0]])
    
    rn=r+tf.tensordot(al_j,fk,1)
    rn=tf.clip_by_value(rn,rmin,rmax)
    return rn 

def comp_Zh(lx,dx,lw=[]):
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

def comp_Zca(lx, a, itraj=[], lw=[], dt=1, strict=False, dx=1e-3, mindx=1e-3, eq=False):
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

def comp_Zca_ekn(ekn,a,dx=None,strict=False,eq=False):
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


@tf.function
def NPNEw(r0,y,It,nr,ny):
    
    r=r0/tf.math.reduce_max(r0)
    fk=[]
    f=tf.ones_like(r)
    for ir in range(ny+1):
        fy=tf.identity(f)
        for iy in range(ny+1-ir):
            fk.append(fy)
            fy=fy*y
        f=f*r
    for ir in range(ny+1,nr+1):
        fk.append(f)
        f=f*r
    fk=tf.stack(fk)
    
    dfk=fk-tf.roll(fk,-1,1)

    b=-tf.tensordot(dfk,r0*It, 1)
    b=tf.reshape(b, [b.shape[0],1])
    scale=tf.math.reduce_sum(1-r0*It)
    scale=tf.reshape(scale,[1,1])
    b=tf.concat((b,scale),0)
    
    ones=tf.reshape(It,[1,It.shape[0]])
    dfk=tf.concat((dfk*It,ones),0)
    akj=tf.tensordot(dfk,fk,axes=[1,1])
    
    al_j=tf.linalg.lstsq(akj,b,fast=False)
    al_j=tf.reshape(al_j, [al_j.shape[0]])
    
    rn=r0+tf.tensordot(al_j,fk,1)
    
    return rn

def tonatural(lq,dx,dtsim=1,itraj=[],lw=[],fixZhA=True,zcm1=False):
#   Zc1=dt*D*Zh; dt=1
#   D=Zc1/(Zh*dt)
#   dy/dx=D^-1/2
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