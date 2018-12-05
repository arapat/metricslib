
const EPSILON: f32 = 1e-8;


#[inline]
pub fn get_sign(a: f32) -> i8 {
    if a < -EPSILON {
        -1
    } else if a > EPSILON {
        1
    } else {
        0
    }
}


#[inline]
pub fn is_zero(a: f32) -> bool {
    get_sign(a as f32) == 0
}
