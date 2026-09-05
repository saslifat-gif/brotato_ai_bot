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
