struct Hit {
    d: f32,
    norm: vec3<f32>,
    color: vec3<f32>,
}

const OUT = Hit(1000000.0, vec3(-1.0), vec3(-1.0));

fn ABox(origin: vec2<f32>, idir: vec2<f32>, bmin: vec2<f32>, bmax: vec2<f32>) -> vec2<f32> {
    //Returns near/far for box
    let tMin = (bmin-origin)*idir;
    let tMax = (bmax-origin)*idir;
    let t1 = min(tMin,tMax);
    let t2 = max(tMin,tMax);
    return vec2(max(t1.x,t1.y),min(t2.x,t2.y));
}

fn ABox3(origin: vec3<f32>, idir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    //Returns near/far for box
    let tMin = (bmin-origin)*idir;
    let tMax = (bmax-origin)*idir;
    let t1 = min(tMin,tMax);
    let t2 = max(tMin,tMax);
    return vec2(max(max(t1.x,t1.y),t1.z),min(min(t2.x,t2.y),t2.z));
}

fn ABoxfar(origin: vec2<f32>, idir: vec2<f32>, bmin: vec2<f32>, bmax: vec2<f32>) -> f32 {
    //Returns far for box
    let tMin = (bmin-origin)*idir;
    let tMax = (bmax-origin)*idir;
    let t2 = max(tMin,tMax);
    return min(t2.x,t2.y);
}

fn ABoxfar3(origin: vec3<f32>, idir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> f32 {
    //Returns far for box
    let tMin = (bmin-origin)*idir;
    let tMax = (bmax-origin)*idir;
    let t2 = max(tMin,tMax);
    return min(min(t2.x,t2.y),t2.z);
}

fn ABoxfarNormal(origin: vec2<f32>, idir: vec2<f32>, bmin: vec2<f32>, bmax: vec2<f32>, dist: ptr<function, f32>) -> vec2<f32> {
    //Returns far normal, far distance as out
    let tMin = (bmin-origin)*idir;
    let tMax = (bmax-origin)*idir;
    let t2 = max(tMin,tMax);
    *dist = min(t2.x,t2.y);
    let signdir = (max(vec2(0.),sign(idir))*2.-1.);
    if (t2.x<t2.y) {
        return vec2(signdir.x,0.);
    } else {
        return vec2(0.,signdir.y);
    }
}

fn ABoxfarNormal3(origin: vec3<f32>, idir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>, dist: ptr<function, f32>) -> vec3<f32> {
    //Returns far normal, far distance as out
    let tMin = (bmin-origin)*idir;
    let tMax = (bmax-origin)*idir;
    let t2 = max(tMin,tMax);
    *dist = min(min(t2.x,t2.y),t2.z);
    let signdir = (max(vec3(0.),sign(idir))*2.-1.);
    if (t2.x<min(t2.y,t2.z)) {
        return vec3(signdir.x,0.,0.);
    } else if (t2.y<t2.z) {
        return vec3(0.,signdir.y,0.);
    } else {
        return vec3(0.,0.,signdir.z);
    }
}

fn ABoxNormal(origin: vec3<f32>, idir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>, n: ptr<function, vec3<f32>>) -> vec2<f32> {
    //Returns near/far, near normal as out
    let tMin = (bmin-origin)*idir;
    let tMax = (bmax-origin)*idir;
    let t1 = min(tMin,tMax);
    let t2 = max(tMin,tMax);
    let signdir = -(max(vec3(0.),sign(idir))*2.-1.);
    if (t1.x>max(t1.y,t1.z)) {
        *n = vec3(signdir.x,0.,0.);
    } else if (t1.y>t1.z) {
        *n = vec3(0.,signdir.y,0.);
    } else {
        *n = vec3(0.,0.,signdir.z);
    }
    return vec2(max(max(t1.x,t1.y),t1.z),min(min(t2.x,t2.y),t2.z));
}

fn ABoxNormal3(origin: vec3<f32>, idir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec3<f32> {
    //Returns near normal
    let tMin = (bmin-origin)*idir;
    let tMax = (bmax-origin)*idir;
    let t1 = min(tMin,tMax);
    let t2 = max(tMin,tMax);
    let signdir = -(max(vec3(0.),sign(idir))*2.-1.);
    if (t1.x>max(t1.y,t1.z)) {
        return vec3(signdir.x,0.,0.);
    } else if (t1.y>t1.z) {
        return vec3(0.,signdir.y,0.);
    } else {
        return vec3(0.,0.,signdir.z);
    }
}

fn ABoxXZ(P: vec3<f32>, D: vec3<f32>, BMin: vec3<f32>, BSize: vec3<f32>, XP: vec3<f32>, XN: vec3<f32>, ZP: vec3<f32>, ZN: vec3<f32>) -> vec4<f32> {
    //Returns the nearest distance to the twisted box (flat horisonal surfaces)
    let PXP = BMin+BSize*vec3(1.,0.5,0.5);
    let PXN = BMin+BSize*vec3(0.,0.5,0.5);
    let PZP = BMin+BSize*vec3(0.5,0.5,1.);
    let PZN = BMin+BSize*vec3(0.5,0.5,0.);
    let dots = vec4(dot(XP,D),dot(XN,D),dot(ZP,D),dot(ZN,D));
    let dists = vec4(-dot(P-PXP,XP)/dots.x, -dot(P-PXN,XN)/dots.y, -dot(P-PZP,ZP)/dots.z, -dot(P-PZN,ZN)/dots.w);
    var N: vec3<f32>;
    if (dots.x<0.) {
        let ip = P+D*dists.x;
        if (max(dot(ip-PXN,XN),max(dot(ip-PZP,ZP),dot(ip-PZN,ZN)))<0. && ip.y>BMin.y && ip.y<BMin.y+BSize.y) {
            return vec4(normalize(XP),dists.x);
        }
    }
    if (dots.y<0.) {
        let ip = P+D*dists.y;
        if (max(dot(ip-PXP,XP),max(dot(ip-PZP,ZP),dot(ip-PZN,ZN)))<0. && ip.y>BMin.y && ip.y<BMin.y+BSize.y) {
            return vec4(normalize(XN),dists.y);
        }
    }
    if (dots.z<0.) {
        let ip = P+D*dists.z;
        if (max(dot(ip-PZN,ZN),max(dot(ip-PXP,XP),dot(ip-PXN,XN)))<0. && ip.y>BMin.y && ip.y<BMin.y+BSize.y) {
            return vec4(normalize(ZP),dists.z);
        }
    }
    if (dots.w<0.) {
        let ip = P+D*dists.w;
        if (max(dot(ip-PZP,ZP),max(dot(ip-PXP,XP),dot(ip-PXN,XN)))<0. && ip.y>BMin.y && ip.y<BMin.y+BSize.y) {
            return vec4(normalize(ZN),dists.w);
        }
    }
    //No hit
    return vec4(-1.);
}

fn ABoxXZTopBottom(P: vec3<f32>, D: vec3<f32>, BMin: vec3<f32>, BSize: vec3<f32>, XP: vec3<f32>, XN: vec3<f32>, ZP: vec3<f32>, ZN: vec3<f32>) -> vec4<f32> {
    //Returns the nearest distance to the twisted box (flat horisonal surfaces)
    let PXP = BMin+BSize*vec3(1.,0.5,0.5);
    let PXN = BMin+BSize*vec3(0.,0.5,0.5);
    let PZP = BMin+BSize*vec3(0.5,0.5,1.);
    let PZN = BMin+BSize*vec3(0.5,0.5,0.);
    let dots = vec4(dot(XP,D),dot(XN,D),dot(ZP,D),dot(ZN,D));
    let dists = vec4(-dot(P-PXP,XP)/dots.x, -dot(P-PXN,XN)/dots.y, -dot(P-PZP,ZP)/dots.z, -dot(P-PZN,ZN)/dots.w);
    var N: vec3<f32>;
    if (dots.x<0.) {
        let ip = P+D*dists.x;
        if (max(dot(ip-PXN,XN),max(dot(ip-PZP,ZP),dot(ip-PZN,ZN)))<0. && ip.y>BMin.y && ip.y<BMin.y+BSize.y) {
            return vec4(normalize(XP),dists.x);
        }
    }
    if (dots.y<0.) {
        let ip = P+D*dists.y;
        if (max(dot(ip-PXP,XP),max(dot(ip-PZP,ZP),dot(ip-PZN,ZN)))<0. && ip.y>BMin.y && ip.y<BMin.y+BSize.y) {
            return vec4(normalize(XN),dists.y);
        }
    }
    if (dots.z<0.) {
        let ip = P+D*dists.z;
        if (max(dot(ip-PZN,ZN),max(dot(ip-PXP,XP),dot(ip-PXN,XN)))<0. && ip.y>BMin.y && ip.y<BMin.y+BSize.y) {
            return vec4(normalize(ZP),dists.z);
        }
    }
    if (dots.w<0.) {
        let ip = P+D*dists.w;
        if (max(dot(ip-PZP,ZP),max(dot(ip-PXP,XP),dot(ip-PXN,XN)))<0. && ip.y>BMin.y && ip.y<BMin.y+BSize.y) {
            return vec4(normalize(ZN),dists.w);
        }
    }
    //Flat surfaces
    var rpy = P.y-BMin.y-BSize.y;
    if (max(-rpy,D.y)<0.) {
        let ip = P-D*(rpy/D.y);
        if (max(max(dot(ip-PXP,XP),dot(ip-PXN,XN)),max(dot(ip-PZP,ZP),dot(ip-PZN,ZN)))<0.) {
            return vec4(0.,1.,0., -rpy/D.y);
        }
    }
    rpy = P.y-BMin.y;
    if (max(rpy, -D.y)<0.) {
        let ip = P-D*(rpy/D.y);
        if (max(max(dot(ip-PXP,XP),dot(ip-PXN,XN)),max(dot(ip-PZP,ZP),dot(ip-PZN,ZN)))<0.) {
            return vec4(0., -1.,0., -rpy/D.y);
        }
    }
    //No hit
    return vec4(-1.);
}

fn ABoxXY(P: vec3<f32>, D: vec3<f32>, BMin: vec3<f32>, BSize: vec3<f32>, XP: vec3<f32>, XN: vec3<f32>, YP: vec3<f32>, YN: vec3<f32>) -> vec4<f32> {
    //Returns the nearest distance to the twisted box (flat horisonal surfaces)
    let PXP = BMin+BSize*vec3(1.,0.5,0.5);
    let PXN = BMin+BSize*vec3(0.,0.5,0.5);
    let PYP = BMin+BSize*vec3(0.5,1.,0.5);
    let PYN = BMin+BSize*vec3(0.5,0.,0.5);
    let dots = vec4(dot(XP,D),dot(XN,D),dot(YP,D),dot(YN,D));
    let dists = vec4(-dot(P-PXP,XP)/dots.x, -dot(P-PXN,XN)/dots.y, -dot(P-PYP,YP)/dots.z, -dot(P-PYN,YN)/dots.w);
    var N: vec3<f32>;
    if (dots.x<0.) {
        let ip = P+D*dists.x;
        if (max(dot(ip-PXN,XN),max(dot(ip-PYP,YP),dot(ip-PYN,YN)))<0. && ip.z>BMin.z && ip.z<BMin.z+BSize.z) {
            return vec4(normalize(XP),dists.x);
        }
    }
    if (dots.y<0.) {
        let ip = P+D*dists.y;
        if (max(dot(ip-PXP,XP),max(dot(ip-PYP,YP),dot(ip-PYN,YN)))<0. && ip.z>BMin.z && ip.z<BMin.z+BSize.z) {
            return vec4(normalize(XN),dists.y);
        }
    }
    if (dots.z<0.) {
        let ip = P+D*dists.z;
        if (max(dot(ip-PYN,YN),max(dot(ip-PXP,XP),dot(ip-PXN,XN)))<0. && ip.z>BMin.z && ip.z<BMin.z+BSize.z) {
            return vec4(normalize(YP),dists.z);
        }
    }
    if (dots.w<0.) {
        let ip = P+D*dists.w;
        if (max(dot(ip-PYP,YP),max(dot(ip-PXP,XP),dot(ip-PXN,XN)))<0. && ip.z>BMin.z && ip.z<BMin.z+BSize.z) {
            return vec4(normalize(YN),dists.w);
        }
    }
    //No hit
    return vec4(-1.);
}

fn DFBox3(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = abs(p-b*0.5)-b*0.5;
    return min(max(d.x,max(d.y,d.z)),0.)+length(max(d,vec3(0.)));
}

fn DFBoxC3(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = abs(p)-b;
    return min(max(d.x,max(d.y,d.z)),0.)+length(max(d,vec3(0.)));
}

fn DFBox(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = abs(p-b*0.5)-b*0.5;
    return min(max(d.x,d.y),0.)+length(max(d,vec2(0.)));
}

fn DFBoxC(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = abs(p)-b;
    return min(max(d.x,d.y),0.)+length(max(d,vec2(0.)));
}

fn DFDisk(p: vec3<f32>) -> f32 {
    let d = length(p.xz-0.5)-0.35;
    let w = vec2(d,abs(p.y));
    return min(max(w.x,w.y),0.)+length(max(w,vec2(0.)));
}

fn trace(pos: vec3<f32>, dir: vec3<f32>) -> Hit {
    var result = OUT;

    let dir_recip = 1.0 / dir;
    var norm: vec3<f32>;
    var sd: vec4<f32>;
    
    //Flat floor
    if (dir.y < 0.0 && pos.y > 0.0) {
        result = Hit(-pos.y / dir.y, vec3(0.0, 1.0, 0.0), vec3(0.99));
    }
    
    //Wall not window
    var bb = ABoxNormal(pos, dir_recip, vec3(3.0, 0.0, 0.0), vec3(3.1, 3.0, 3.0), &norm);
    if (bb.x > 0.0 && bb.y > bb.x && bb.x < result.d) {
        result = Hit(bb.x, norm, vec3(1.0));
    }

    //Wall window
    bb = ABox3(pos, dir_recip, vec3(0.0, 0.0, 2.8), vec3(3.1, 2.8, 3.1));
    if (DFBox3(pos - vec3(0.0, 0.0, 2.8), vec3(3.1, 2.8, 0.3)) < 0.0 || (bb.x > 0.0 && bb.y > bb.x && bb.x < result.d)) {
        //Emissive windows
        bb = ABoxNormal(pos, dir_recip, vec3(0.05, 0.0, 3.0), vec3(2.9, 2.2, 3.1), &norm);
        var Emissive = mix(mix(vec3(3.0, 2.0, 2.0), vec3(4.0, 0.4, 0.1),
                        f32(pos.x + dir.x * bb.x > 2.0)), vec3(1.3, 1.0, 4.0), f32(pos.x + dir.x * bb.x < 1.0));
        Emissive = mix(Emissive, vec3(1.0), f32(pos.y + dir.y * bb.x > 1.25 && abs(pos.x + dir.x * bb.x - 1.5) < 0.5));
        if (bb.x > 0.0 && bb.y > bb.x && bb.x < result.d) {
            result = Hit(bb.x, norm, Emissive);
        }
        //Window frames
        sd = ABoxXZ(pos, dir, vec3(0.0, 0.0, 2.9), vec3(0.225, 2.8, 0.2),
                        vec3(1.0, -0.1, -0.02), vec3(-1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(-0.4, 0.015, -1.0));
        if (sd.w > 0.0 && sd.w < result.d) {
            result = Hit(sd.w, sd.xyz, vec3(1.0));
        }
        sd = ABoxXZ(pos, dir, vec3(0.9, 0.0, 2.9), vec3(0.3, 2.8, 0.2),
                        vec3(1.0, -0.03, -0.1), vec3(-1.0, -0.02, -0.25), vec3(0.0, 0.0, 1.0), vec3(0.2, 0.015, -1.0));
        if (sd.w > 0.0 && sd.w < result.d) {
            result = Hit(sd.w, sd.xyz, vec3(1.0));
        }
        sd = ABoxXZ(pos, dir, vec3(1.8, 0.0, 2.9), vec3(0.35, 2.8, 0.2),
                        vec3(1.0, 0.04, 0.1), vec3(-1.0, 0.08, 0.25), vec3(0.0, 0.0, 1.0), vec3(0.2, 0.015, -1.0));
        if (sd.w > 0.0 && sd.w < result.d) {
            result = Hit(sd.w, sd.xyz, vec3(1.0));
        }
        sd = ABoxXZ(pos, dir, vec3(2.8, 0.0, 2.9), vec3(0.2, 2.8, 0.2),
                        vec3(1.0, 0.0, 0.0), vec3(-1.0, -0.04, -0.02), vec3(0.0, 0.0, 1.0), vec3(-0.4, 0.015, -1.0));
        if (sd.w > 0.0 && sd.w < result.d) {
            result = Hit(sd.w, sd.xyz, vec3(1.0));
        }
    }

    //Twisted wall over
    bb = ABox3(pos, dir_recip, vec3(0.0, 2.2, 2.0), vec3(3.1, 3.0, 3.1));
    if (DFBox3(pos - vec3(0.0, 2.2, 2.0), vec3(3.1, 0.8, 1.1)) < 0.0 || (bb.x > 0.0 && bb.y > bb.x && bb.x < result.d)) {
        //Wall behind
        sd = ABoxXZTopBottom(pos, dir, vec3(0.0, 2.2, 2.7), vec3(3.0, 0.8, 0.4),
                        vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, -0.707, -1.0));
        if (sd.w > 0.0 && sd.w < result.d) {
            result = Hit(sd.w, sd.xyz, vec3(1.0));
        }
    }
    
    
    
    //Floor boards
    for (var xi = 0.0; xi < 5.0; xi += 1.0) {
        bb = ABox3(pos, dir_recip, vec3(xi * 0.63 - 0.03, 0.0, 0.0), vec3(xi * 0.63 + 0.8, 0.05, 3.1));
        if (DFBox3(pos - vec3(xi * 0.6, 0.0, 0.0), vec3(0.8, 0.05, 3.1)) < 0.0 || (bb.x > 0.0 && bb.y > bb.x && bb.x < result.d)) {
            sd = ABoxXY(pos, dir, vec3(xi * 0.63, 0.0, 0.05), vec3(0.2, 0.03, 3.05),
                            vec3(1.0, 0.8, -0.01), vec3(-1.0, 1.6, -0.01), vec3(-0.01, 1.0, -0.005), vec3(0.0, -1.0, 0.0));
            if (sd.w > 0.0 && sd.w < result.d) {
                result = Hit(sd.w, sd.xyz, vec3(1.0));
            }
            sd = ABoxXY(pos, dir, vec3(xi * 0.63 + 0.21, 0.0, 0.05), vec3(0.2, 0.03, 3.05),
                            vec3(1.0, 0.75, -0.005), vec3(-1.0, 0.75, 0.0), vec3(-0.01, 1.0, -0.005), vec3(0.0, -1.0, 0.0));
            if (sd.w > 0.0 && sd.w < result.d) {
                result = Hit(sd.w, sd.xyz, vec3(1.0));
            }
            sd = ABoxXY(pos, dir, vec3(xi * 0.63 + 0.42, 0.0, 0.05), vec3(0.2, 0.03, 3.05),
                            vec3(1.0, 0.8, 0.0), vec3(-1.0, 0.75, 0.0), vec3(-0.09, 1.0, 0.0025), vec3(0.0, -1.0, 0.0));
            if (sd.w > 0.0 && sd.w < result.d) {
                result = Hit(sd.w, sd.xyz, vec3(1.0));
            }
        }
    }
    
    
    
    //Table
    let rpos = pos - vec3(0.1, 0.0, 1.25);
    bb = ABox3(rpos, dir_recip, vec3(0.0, 0.0, 0.0), vec3(0.75, 0.5, 0.75));
    if (DFBox3(rpos, vec3(0.75, 0.5, 0.75))<0.0 || (bb.x > 0.0 && bb.y > bb.x && bb.x < result.d)) {
        //Legs
        sd = ABoxXZ(rpos, dir, vec3(0.15, 0.0, 0.15), vec3(0.05, 0.45, 0.05),
                    vec3(1.0, 0.07, -0.15), vec3(-1.0, 0.03, 0.3), vec3(-0.5, 0.02, 1.0), vec3(0.3, 0.07, -1.0));
        if (sd.w > 0.0 && sd.w < result.d) {
            result = Hit(sd.w, sd.xyz, vec3(1.0));
        }
        sd = ABoxXZ(rpos, dir, vec3(0.55, 0.0, 0.325), vec3(0.05, 0.45, 0.05),
                    vec3(1.0, -0.07, -0.15), vec3(-1.0, -0.01, 0.3), vec3(-0.5, -0.02, 1.0), vec3(0.3, 0.07, -1.0));
        if (sd.w > 0.0 && sd.w < result.d) {
            result = Hit(sd.w, sd.xyz, vec3(1.0));
        }
        sd = ABoxXZ(rpos, dir, vec3(0.25, 0.0, 0.55), vec3(0.05, 0.45, 0.05),
                    vec3(1.0, 0.01, -0.15), vec3(-1.0, 0.01, 0.3), vec3(-0.5, 0.02, 1.0), vec3(0.3, -0.07, -1.0));
        if (sd.w > 0.0 && sd.w < result.d) {
            result = Hit(sd.w, sd.xyz, vec3(1.0));
        }
        //Table
        sd = ABoxXZTopBottom(rpos, dir, vec3(0.07, 0.45, 0.1), vec3(0.6, 0.025, 0.55),
                    vec3(1.0, 0.8, 0.15), vec3(-1.0, 0.8, 0.5), vec3(0.6, 0.8, 1.0), vec3(-0.1, 0.8, -1.0));
        if (sd.w > 0.0 && sd.w < result.d) {
            result = Hit(sd.w, sd.xyz, vec3(1.0));
        }
    }
    
    
    
    //Boxes
    sd = ABoxXZTopBottom(pos, dir, vec3(0.05, 0.0, 2.3), vec3(0.45),
                vec3(1.0, 0.0, -0.6), vec3(-1.0, 0.0, 0.6), vec3(0.6, 0.0, 1.0), vec3(-0.6, 0.0, -1.0));
    if (sd.w > 0.0 && sd.w < result.d) {
        result = Hit(sd.w, sd.xyz, vec3(1.0));
    }
    sd = ABoxXZTopBottom(pos, dir, vec3(-0.1, 0.45, 2.2), vec3(0.7, 0.45, 0.7),
                vec3(1.0, 0.0, -1.6), vec3(-1.0, 0.0, 1.6), vec3(1.6, 0.0, 1.0), vec3(-1.6, 0.0, -1.0));
    if (sd.w > 0.0 && sd.w < result.d) {
        result = Hit(sd.w, sd.xyz, vec3(1.0));
    }
    
    //Wall box
    bb = ABoxNormal(pos, dir_recip, vec3(2.8, 1.0, 1.0), vec3(3.0, 1.05, 2.0), &norm);
    if (bb.x > 0.0 && bb.y > bb.x && bb.x < result.d) {
        result = Hit(bb.x, norm, vec3(1.0));
    }
    bb = ABoxNormal(pos, dir_recip, vec3(2.8, 1.3, 1.1), vec3(3.0, 1.35, 2.1), &norm);
    if (bb.x > 0.0 && bb.y > bb.x && bb.x < result.d) {
        result = Hit(bb.x, norm, vec3(1.0));
    }
    bb = ABoxNormal(pos, dir_recip, vec3(2.8, 1.65, 1.2), vec3(3.0, 1.7, 2.3), &norm);
    if (bb.x > 0.0 && bb.y > bb.x && bb.x < result.d) {
        result = Hit(bb.x, norm, vec3(1.0));
    }
    bb = ABoxNormal(pos, dir_recip, vec3(2.85, 1.0, 1.575), vec3(3.0, 1.65, 1.625), &norm);
    if (bb.x > 0.0 && bb.y > bb.x && bb.x < result.d) {
        result = Hit(bb.x, norm, vec3(1.0));
    }
    
    
    
    //Wall cupb
    sd = ABoxXZTopBottom(pos, dir, vec3(2.6, 2.0, 0.4), vec3(0.4, 0.6, 0.4),
                vec3(1.0, 0.0, 0.0), vec3(-1.0, -0.3, 0.0), vec3(0.0, -0.05, 1.0), vec3(0.0, -0.15, -1.0));
    if (sd.w > 0.0 && sd.w < result.d) {
        result = Hit(sd.w, sd.xyz, vec3(1.0));
    }
    
    
    
    //Desk
    bb = ABoxNormal(pos, dir_recip, vec3(2.4, 0.6, 1.0), vec3(3.0, 0.65, 2.5), &norm);
    if (bb.x > 0.0 && bb.y > bb.x && bb.x < result.d) {
        result = Hit(bb.x, norm, vec3(1.0));
    }
    sd = ABoxXZTopBottom(pos, dir, vec3(2.5, 0.0, 1.1), vec3(0.5, 0.6, 0.3),
            vec3(1.0, 0.0, 0.0), vec3(-1.0, -0.3, 0.0), vec3(0.0, -0.05, 1.0), vec3(0.0, -0.15, -1.0));
    if (sd.w > 0.0 && sd.w < result.d) {
        result = Hit(sd.w, sd.xyz, vec3(1.0));
    }
    sd = ABoxXZTopBottom(pos, dir, vec3(2.45, 0.0, 2.35), vec3(0.1, 0.6, 0.05),
            vec3(1.0, 0.0, 0.0), vec3(-1.0, -0.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0));
    if (sd.w > 0.0 && sd.w < result.d) {
        result = Hit(sd.w, sd.xyz, vec3(1.0));
    }
    
    
    
    //Textured vertical planes
    if (sign(dir.z * (pos.z - 0.9)) < 0.0) {
        let t = -(pos.z - 0.9) / dir.z;
        let sp = pos + dir * t;
        if (t < result.d && DFBox(sp.xy - vec2(2.5, 0.0), vec2(0.5, 1.7))<0.0 &&
            (sin((sp.x - 2.5) * 27.0 - 1.9) * 0.5 + 0.5)*(sin(sp.y * 27.0) * 0.5 + 0.5)<0.5) {
            result = Hit(t, vec3(0.0, 0.0, -sign(dir.z)), vec3(1.0));
        }
    }
    if (sign(dir.z * (pos.z - 0.3)) < 0.0) {
        let t = -(pos.z - 0.3) / dir.z;
        let sp = pos + dir * t;
        if (t < result.d && DFBox(sp.xy - vec2(2.5, 0.0), vec2(0.5, 1.7)) < 0.0 &&
            (sin((sp.x - 2.5) * 27.0 - 1.9) * 0.5 + 0.5)*(sin(sp.y * 27.0) * 0.5 + 0.5) < 0.5) {
            result = Hit(t, vec3(0.0, 0.0, -sign(dir.z)), vec3(1.0));
        }
    }
    
    
    
    //Textured wall plane
    if (dir.x > 0.0 && pos.x < 2.975) {
        let t = -(pos.x - 2.975) / dir.x;
        let sp = pos + dir * t;
        if (t < result.d && DFBox(sp.zy, vec2(3.0)) < 0.0 &&
            (sin(sp.z * 33.0 - 1.0) * 0.5 + 0.5) * (sin(sp.y * 33.0 - 0.25) * 0.5 + 0.5) < 0.4) {
            result = Hit(t, vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.4, 0.05));
        }
    }
    
    
    
    //Texture floor plane
    if (dir.y < 0.0) {
        let t = -(pos.y - 0.06) / dir.y;
        let sp = pos + dir * t;
        if (t < result.d && DFBox(sp.xz - vec2(1.0, 0.325), vec2(1.0, 2.5)) < 0.0 &&
            (sin(sp.x * 33.0) * 0.5 + 0.5) * (sin(sp.z * 33.0) * 0.5 + 0.5) < 0.4) {
            let z = (pos.z + dir.z * t) * 0.34;
            result = Hit(t, vec3(0.0, 1.0, 0.0), mix(vec3(1.0), vec3(0.1, 1.0, 0.1), z * z * (3.0 - 2.0 * z)));
        }
    }

    return result;
}

fn tbn(norm: vec3<f32>) -> mat3x3<f32> {
    //Returns the simple tangent space matrix
    var normb: vec3<f32>;
    var normt: vec3<f32>;
    if (abs(norm.y)>0.999) {
        normb = vec3(1.0, 0.0, 0.0);
        normt = vec3(0.0, 0.0, 1.0);
    } else {
    	normb = normalize(cross(norm,vec3(0.0, 1.0, 0.0)));
    	normt = normalize(cross(normb,norm));
    }
    return mat3x3(normb.x,normt.x,norm.x,normb.y,normt.y,norm.y,normb.z,normt.z,norm.z);
}

fn vec3ToF32(v: vec3<f32>) -> f32 {
    //Returns "int" from vec3 (10 bit per channel)
    let intv = min(vec3<i32>(floor(v*1024.0)), vec3(1023i));
    return bitcast<f32>(intv.x + intv.y * 1024 + intv.z * 1048576);
}

const FOV = 0.6;
const CFOV = tan(FOV);
const LIGHT_COEFF = 10.0;
const LIGHT_COEFF_RECIP = 1.0 / LIGHT_COEFF;

@group(0)
@binding(0)
var image: texture_storage_2d<rgba32float, read_write>;
@group(0)
@binding(1)
var<uniform> size: vec2<u32>;

@compute
@workgroup_size(16, 8)
fn renderGbuffer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= size.x || global_id.y >= size.y {
        return;
    }

    let size = vec2<f32>(size);
    let eye = vec3(0.9, -0.5, 0.6);

    let pos = vec3(1.5, 1.0,1.75) - eye * 4.0;
    let eye_mat = tbn(eye);
    let dir = normalize(vec3((vec2<f32>(global_id.xy) / size * 2.0 - 1.0) * vec2(1.0, -1.0) * (size.x / size.y * CFOV), 1.0) * eye_mat);
    
    let pixel = trace(pos, dir);
    var result: vec4<f32>;

    if (pixel.color.r > 1.0) {
            //Emissive
            result = vec4(vec3ToF32(vec3(pixel.color.r - 1.0, pixel.color.yz) * LIGHT_COEFF_RECIP), -1.0, vec3ToF32(pixel.norm * 0.5 + 0.5), pixel.d);
        } else if (pixel.color.r > -0.5) {
            //Geometry
            let ppos = pos + dir * pixel.d + pixel.norm * 0.0001;

            result = vec4(
                0.0,
                vec3ToF32(pixel.color),
                vec3ToF32(pixel.norm * 0.5 + 0.5),
                pixel.d,
            );
        } else {
            //Sky
            result = vec4(0.0, -2.0, 0.0, 100000.0);
        }

    textureStore(image, global_id.xy, result);
}

@vertex
fn displayVertex(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2(
        f32((i32(vertex_index) << 1u) & 2),
        f32(i32(vertex_index) & 2),
    );

    let pos = 2.0 * uv - vec2(1.0, 1.0);

    return vec4(pos.x, pos.y, 0.0, 1.0);
}

@fragment
fn displayFragment(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(image, vec2<u32>(pos.xy));
}