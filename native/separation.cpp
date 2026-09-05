#include <cmath>
#include <algorithm>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
// Rows: enemy x,y,vx,vy,radius. Output: index,current,predicted,target,closing,radial.
API void separation(const double* e, int n, double px, double py, double pr,
                    double speed, double range, int ranged, double horizon, double* out) {
 const double q=std::sqrt(.5);
 const double a[9][2]={{0,0},{0,-1},{0,1},{-1,0},{1,0},{-q,-q},{q,-q},{-q,q},{q,q}};
 for(int k=0;k<9;++k){
  double* o=out+k*6; o[0]=-1; for(int z=1;z<6;++z)o[z]=0;
  for(int i=0;i<n;++i){
   const double* r=e+5*i;
   double d=std::hypot(r[0]+r[2]*horizon-(px+a[k][0]*speed*horizon),r[1]+r[3]*horizon-(py+a[k][1]*speed*horizon));
   double c=std::hypot(r[0]-px,r[1]-py);
   if(o[0]>=0 && !(d<o[2] || (d==o[2] && c<o[1])))continue;
   double t=pr+r[4]+80;
   if(ranged)t=std::max(t,std::min(420.,std::max(180.,range*.55)));
   o[0]=i;o[1]=c;o[2]=d;o[3]=t;
   o[4]=d>t*1.4?0:std::max(0.,c-d)/std::max(1.,t);
   o[5]=a[k][0]*(px-r[0])/std::max(1.,c)+a[k][1]*(py-r[1])/std::max(1.,c);
  }
 }
}
API double crowd(const double* e,int n,double fx,double fy,double ax,double ay,double px,double py){
 int count=0;double sx=0,sy=0;
 for(int i=0;i<n;++i){const double* r=e+5*i;double x=r[0]+r[2]*.45,y=r[1]+r[3]*.45;
  if(std::hypot(x-fx,y-fy)>=240.)continue;count++;sx+=x;sy+=y;}
 if(count<=1)return 0.;
 double density=std::min(4.,std::pow(count/6.,2)*.9),mx=sx/count-px,my=sy/count-py,len=std::max(1.,std::hypot(mx,my));
 return density*(1.+.6*std::max(0.,ax*mx/len+ay*my/len));
}
// Hazard rows x,y,vx,vy,expanded radius. Coin rows x,y.
API double route(const double* e,int n,const double* coins,int nc,double px,double py,double tx,double ty,double width,double height,double speed){
 if(!(100.<=tx && tx<=width-100. && 100.<=ty && ty<=height-100.))return -1.;
 double distance=std::hypot(tx-px,ty-py),travel=distance/speed,pressure=0.;
 for(int i=0;i<n;++i){const double* r=e+5*i;double rx=r[0]-px,ry=r[1]-py,dx=r[2]*travel-(tx-px),dy=r[3]*travel-(ty-py);
  double start=std::hypot(rx,ry)-r[4],length2=dx*dx+dy*dy;
  double closest=length2?std::max(0.,std::min(1.,-(rx*dx+ry*dy)/length2)):0.;
  if(std::hypot(rx+dx*closest,ry+dy*closest)-r[4]<std::min(20.,start)-1e-6)return -1.;
  for(double f:{.25,.5,.75,1.}){
   double c=std::hypot(r[0]+r[2]*travel*f-(px+(tx-px)*f),r[1]+r[3]*travel*f-(py+(ty-py)*f))-r[4];
   if(c<std::min(20.,start+10.))return -1.;pressure+=std::max(0.,220.-c)/220.*(f<1.?.5:1.);
  }
 }
 double edge=std::min(std::min(tx,ty),std::min(width-tx,height-ty));pressure+=2.*std::max(0.,240.-edge)/240.;
 int count=0;for(int i=0;i<nc;++i)if(std::hypot(coins[i*2]-tx,coins[i*2+1]-ty)<120.)count++;
 return pressure-std::min(.12,count*.02)+distance/10000.;
}
// Each result row is distance followed by progress for all nine actions.
API void coin_progress(const double* coins,int n,double px,double py,int shorten,double* out){
 const double q=std::sqrt(.5),a[9][2]={{0,0},{0,-1},{0,1},{-1,0},{1,0},{-q,-q},{q,-q},{-q,q},{q,q}};
 for(int i=0;i<n;++i){double dx=coins[i*2]-px,dy=coins[i*2+1]-py,d=std::hypot(dx,dy),step=shorten?std::min(60.,d):60.;out[i*10]=d;
  for(int k=0;k<9;++k)out[i*10+1+k]=step>0?(d-std::hypot(dx-step*a[k][0],dy-step*a[k][1]))/step:0.;
 }
}
