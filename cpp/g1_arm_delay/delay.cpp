// Bounded causal nominal arm propagation. No SDK, networking or robot output.
// ABI 1: row-major base [acc3, omega3, alpha3, R9], command [ff5,q5,dq5,weight].
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Context {
  mjModel* m = nullptr;
  mjData* d = nullptr;
  std::array<int,5> qi{}, vi{};
  std::array<double,3> offset{};
  std::array<double,9> mount{};
  std::vector<double> mass;
  ~Context() { if(d) mj_deleteData(d); if(m) mj_deleteModel(m); }
};
void cross(const double* a, const double* b, double* out) {
  out[0]=a[1]*b[2]-a[2]*b[1]; out[1]=a[2]*b[0]-a[0]*b[2]; out[2]=a[0]*b[1]-a[1]*b[0];
}
bool finite(const double* p,int n) { for(int i=0;i<n;++i) if(!std::isfinite(p[i])) return false; return true; }
void dynamics(Context& c,const double* q,const double* v,const double* b,double* M,double* bias) {
  auto* m=c.m; auto* d=c.d;
  double root[9]{},r[3]{},wv[3],av[3],wwv[3];
  for(int i=0;i<3;++i) for(int j=0;j<3;++j)
    for(int k=0;k<3;++k) root[3*i+j]+=b[9+3*i+k]*c.mount[3*j+k];
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) r[i]+=root[3*i+j]*c.offset[j];
  cross(b+3,r,wv);cross(b+6,r,av);cross(b+3,wv,wwv);
  mju_copy(d->qpos,m->qpos0,m->nq);
  mju_zero(d->qpos,3); mju_mat2Quat(d->qpos+3,root);
  mju_zero(d->qvel,m->nv);mju_zero(d->qacc,m->nv);
  for(int i=0;i<3;++i) {
    d->qvel[i]=-wv[i];d->qacc[i]=b[i]-av[i]-wwv[i];
    for(int j=0;j<3;++j) { d->qvel[3+i]+=root[3*j+i]*b[3+j];d->qacc[3+i]+=root[3*j+i]*b[6+j]; }
  }
  for(int i=0;i<5;++i) {d->qpos[c.qi[i]]=q[i];d->qvel[c.vi[i]]=v[i];}
  mj_kinematics(m,d);mj_comPos(m,d);mj_crb(m,d);mj_makeM(m,d);
  mj_comVel(m,d);mj_rne(m,d,0,d->qfrc_bias);mj_fullM(m,c.mass.data(),d->qM);
  for(int i=0;i<5;++i) {
    bias[i]=d->qfrc_bias[c.vi[i]];
    for(int j=0;j<m->nv;++j) bias[i]+=c.mass[c.vi[i]*m->nv+j]*d->qacc[j];
    for(int j=0;j<5;++j) M[i*5+j]=c.mass[c.vi[i]*m->nv+c.vi[j]];
  }
}
}
extern "C" {
int g1_delay_abi() {return 1;}
const char* g1_delay_source_sha256() {return G1_DELAY_SOURCE_SHA256;}
int g1_delay_header_version() {return mjVERSION_HEADER;}
int g1_delay_runtime_version() {return mj_version();}
void* g1_delay_create(const char* xml,const int* qi,const int* vi,const double* offset,const double* mount,
                      char* error,int capacity) {
  try {
    auto c=std::make_unique<Context>();c->m=mj_loadXML(xml,nullptr,error,capacity);
    if(!c->m) return nullptr;
    c->d=mj_makeData(c->m);if(!c->d) throw std::runtime_error("cannot create MuJoCo data");
    for(int i=0;i<5;++i) {
      if(qi[i]<7 || qi[i]>=c->m->nq || vi[i]<6 || vi[i]>=c->m->nv) throw std::runtime_error("invalid arm indices");
      c->qi[i]=qi[i];c->vi[i]=vi[i];
    }
    std::copy(offset,offset+3,c->offset.begin());std::copy(mount,mount+9,c->mount.begin());
    c->mass.resize(c->m->nv*c->m->nv);return c.release();
  } catch(const std::exception& e) {std::snprintf(error,capacity,"%s",e.what());return nullptr;}
}
void g1_delay_destroy(void* context) {delete static_cast<Context*>(context);}
int g1_delay_predict(void* context,int count,const double* dt,const double* bases,const double* commands,
                     const unsigned char* present,const double* kp,const double* kd,const double* limits,
                     const double* initial,double* output) {
  if(!context || count<1 || count>40 || !finite(initial,10) || !finite(bases,count*18)
     || !finite(commands,count*16) || !finite(kp,5) || !finite(kd,5) || !finite(limits,5)) return 1;
  auto& c=*static_cast<Context*>(context);std::copy(initial,initial+10,output);
  double startup[5]{},M[25],bias[5],rhs[5],ddq[5];
  for(int step=0;step<count;++step) {
    if(!std::isfinite(dt[step]) || dt[step]<=0 || dt[step]>.002000001) return 2;
    const double* cmd=commands+step*16;
    if(present[step]>1 || cmd[15]<0 || cmd[15]>1) return 2;
    dynamics(c,output,output+5,bases+step*18,M,bias);
    if(step==0) std::copy(bias,bias+5,startup);
    for(int i=0;i<5;++i) {
      if(limits[i]<=0) return 2;
      double tau=present[step] ? cmd[i]+kp[i]*(cmd[5+i]-output[i])+kd[i]*(cmd[10+i]-output[5+i])
                              : startup[i]+kp[i]*(initial[i]-output[i])+kd[i]*(initial[5+i]-output[5+i]);
      if(present[step]) tau=cmd[15]*tau+(1-cmd[15])*bias[i];
      rhs[i]=std::clamp(tau,-limits[i],limits[i])-bias[i];
    }
    if(!finite(M,25) || !finite(rhs,5) || mju_cholFactor(M,5,1e-14)!=5) return 3;
    mju_cholSolve(ddq,M,rhs,5);
    for(int i=0;i<5;++i) {output[i]+=output[5+i]*dt[step]+.5*ddq[i]*dt[step]*dt[step];output[5+i]+=ddq[i]*dt[step];}
    if(!finite(output,10)) return 4;
  }
  return 0;
}
}
