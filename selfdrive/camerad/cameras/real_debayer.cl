#ifdef HALF_AS_FLOAT
#define half float
#define half3 float3
#else
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// post wb CCM
const __constant half3 color_correction_0 = (half3)(1.82717181, -0.31231438, 0.07307673);
const __constant half3 color_correction_1 = (half3)(-0.5743977, 1.36858544, -0.53183455);
const __constant half3 color_correction_2 = (half3)(-0.25277411, -0.05627105, 1.45875782);

half3 color_correct(half3 rgb) {
  half3 ret = (half3)(0.0, 0.0, 0.0);
  ret += (half)rgb.x * color_correction_0;
  ret += (half)rgb.y * color_correction_1;
  ret += (half)rgb.z * color_correction_2;
  return ret;
}

half3 srgb_gamma(half3 p) {
  // go all out and add an sRGB gamma curve
  const half3 ph = (1.0 + 0.055)*pow(p, 1/2.4) - 0.055;
       const half3 pl = p*12.92;
       return select(ph, pl, islessequal(p, 0.0031308));
}

float decompress_12(uint major, uint minor) {
  // decompress - Legacy kneepoints
  float kn_0 = major + minor;
  float kn_2048 = (kn_0 - 2048) * 64 + 2048;
  float kn_3040 = (kn_0 - 3040) * 1024 + 65536;
  return max(kn_0, max(kn_2048, kn_3040));
}

half tonemap(float input, half geometric_mean) {
  // https://www.cl.cam.ac.uk/teaching/1718/AdvGraph/06_HDR_and_tone_mapping.pdf
  // Power function (slide 15)
  // half percentile_99 = 8704.0;
  // half pv = pow(decompressed / percentile_99, (half)0.6) * 0.50;

  // Sigmoidal tone mapping (slide 30)

  // Optimized case of b = 1
  float decompressed_times_a = input * 0.4;
  float pv = decompressed_times_a / (geometric_mean + decompressed_times_a);

  // half b = 1.0;
  // float pow_b = pow(decompressed, b);
  // float pv = pow_b / (pow(geometric_mean / a, b) + pow_b);

  return pv;
}


inline half val_from_10(const uchar * source, int gx, int gy, half black_level, half geometric_mean) {
  // parse 12bit
  int start = gy * FRAME_STRIDE + (3 * (gx / 2)) + (FRAME_STRIDE * FRAME_OFFSET);
  int offset = gx % 2;

  float decompressed = decompress_12((uint)source[start + offset] << 4, (source[start + 2] >> (4 * offset)) & 0xf);
  decompressed -= black_level * 4;
  half pv = tonemap(decompressed, geometric_mean);

  // Original (non HDR)
  // half pv = decompressed / 4096.0;

  pv = clamp(pv, (half)0.0, (half)1.0);
  return pv;
}

half fabs_diff(half x, half y) {
  return fabs(x-y);
}

half phi(half x) {
  // detection funtion
  return 2 - x;
  // if (x > 1) {
  //   return 1 / x;
  // } else {
  //   return 2 - x;
  // }
}

__kernel void debayer10(const __global uchar * in,
                        __global uchar * out,
                        __local half * cached,
                        float black_level,
                        float geometric_mean
                       )
{
  const int x_global = get_global_id(0);
  const int y_global = get_global_id(1);

  const int localRowLen = 2 + get_local_size(0); // 2 padding
  const int x_local = get_local_id(0); // 0-15
  const int y_local = get_local_id(1); // 0-15
  const int localOffset = (y_local + 1) * localRowLen + x_local + 1; // max 18x18-1

  int out_idx = 3 * x_global + 3 * y_global * RGB_WIDTH;

  half pv = val_from_10(in, x_global, y_global, black_level, geometric_mean);
  cached[localOffset] = pv;

  // cache padding
  int localColOffset = -1;
  int globalColOffset = -1;

  const int x_global_mod = (x_global == 0 || x_global == RGB_WIDTH - 1) ? -1: 1;
  const int y_global_mod = (y_global == 0 || y_global == RGB_HEIGHT - 1) ? -1: 1;

  // cache padding
  if (x_local < 1) {
    localColOffset = x_local;
    globalColOffset = -1;
    cached[(y_local + 1) * localRowLen + x_local] = val_from_10(in, x_global-x_global_mod, y_global, black_level, geometric_mean);
  } else if (x_local >= get_local_size(0) - 1) {
    localColOffset = x_local + 2;
    globalColOffset = 1;
    cached[localOffset + 1] = val_from_10(in, x_global+x_global_mod, y_global, black_level, geometric_mean);
  }

  if (y_local < 1) {
    cached[y_local * localRowLen + x_local + 1] = val_from_10(in, x_global, y_global-y_global_mod, black_level, geometric_mean);
    if (localColOffset != -1) {
      cached[y_local * localRowLen + localColOffset] = val_from_10(in, x_global+(x_global_mod*globalColOffset), y_global-y_global_mod, black_level, geometric_mean);
    }
  } else if (y_local >= get_local_size(1) - 1) {
    cached[(y_local + 2) * localRowLen + x_local + 1] = val_from_10(in, x_global, y_global+y_global_mod, black_level, geometric_mean);
    if (localColOffset != -1) {
      cached[(y_local + 2) * localRowLen + localColOffset] = val_from_10(in, x_global+(x_global_mod*globalColOffset), y_global+y_global_mod, black_level, geometric_mean);
    }
  }

  // sync
  barrier(CLK_LOCAL_MEM_FENCE);

  half d1 = cached[localOffset - localRowLen - 1];
  half d2 = cached[localOffset - localRowLen + 1];
  half d3 = cached[localOffset + localRowLen - 1];
  half d4 = cached[localOffset + localRowLen + 1];
  half n1 = cached[localOffset - localRowLen];
  half n2 = cached[localOffset + 1];
  half n3 = cached[localOffset + localRowLen];
  half n4 = cached[localOffset - 1];

  half3 rgb;

  // a simplified version of https://opensignalprocessingjournal.com/contents/volumes/V6/TOSIGPJ-6-1/TOSIGPJ-6-1.pdf
  if (x_global % 2 == 0) {
    if (y_global % 2 == 0) {
      rgb.y = pv; // G1(R)
      half k1 = phi(fabs_diff(d1, pv) + fabs_diff(d2, pv));
      half k2 = phi(fabs_diff(d2, pv) + fabs_diff(d4, pv));
      half k3 = phi(fabs_diff(d3, pv) + fabs_diff(d4, pv));
      half k4 = phi(fabs_diff(d1, pv) + fabs_diff(d3, pv));
      // R_G1
      rgb.x = (k2*n2+k4*n4)/(k2+k4);
      // B_G1
      rgb.z = (k1*n1+k3*n3)/(k1+k3);
    } else {
      rgb.z = pv; // B
      half k1 = phi(fabs_diff(d1, d3) + fabs_diff(d2, d4));
      half k2 = phi(fabs_diff(n1, n4) + fabs_diff(n2, n3));
      half k3 = phi(fabs_diff(d1, d2) + fabs_diff(d3, d4));
      half k4 = phi(fabs_diff(n1, n2) + fabs_diff(n3, n4));
      // G_B
      rgb.y = (k1*(n1+n3)*0.5+k3*(n2+n4)*0.5)/(k1+k3);
      // R_B
      rgb.x = (k2*(d2+d3)*0.5+k4*(d1+d4)*0.5)/(k2+k4);
    }
  } else {
    if (y_global % 2 == 0) {
      rgb.x = pv; // R
      half k1 = phi(fabs_diff(d1, d3) + fabs_diff(d2, d4));
      half k2 = phi(fabs_diff(n1, n4) + fabs_diff(n2, n3));
      half k3 = phi(fabs_diff(d1, d2) + fabs_diff(d3, d4));
      half k4 = phi(fabs_diff(n1, n2) + fabs_diff(n3, n4));
      // G_R
      rgb.y = (k1*(n1+n3)*0.5+k3*(n2+n4)*0.5)/(k1+k3);
      // B_R
      rgb.z = (k2*(d2+d3)*0.5+k4*(d1+d4)*0.5)/(k2+k4);
    } else {
      rgb.y = pv; // G2(B)
      half k1 = phi(fabs_diff(d1, pv) + fabs_diff(d2, pv));
      half k2 = phi(fabs_diff(d2, pv) + fabs_diff(d4, pv));
      half k3 = phi(fabs_diff(d3, pv) + fabs_diff(d4, pv));
      half k4 = phi(fabs_diff(d1, pv) + fabs_diff(d3, pv));
      // R_G2
      rgb.x = (k1*n1+k3*n3)/(k1+k3);
      // B_G2
      rgb.z = (k2*n2+k4*n4)/(k2+k4);
    }
  }

  rgb = clamp(rgb, 0.0, 1.0);
  rgb = color_correct(rgb);
  rgb = srgb_gamma(rgb);
  rgb = clamp(rgb*255.0, 0.0, 255.0);

  out[out_idx + 0] = (uchar)(rgb.z);
  out[out_idx + 1] = (uchar)(rgb.y);
  out[out_idx + 2] = (uchar)(rgb.x);
}
