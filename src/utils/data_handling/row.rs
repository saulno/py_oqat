use std::fmt;

use ordered_float::OrderedFloat;

use super::named_values_set_list::NamedValuesSetList;

use pyo3::prelude::*;

#[derive(Clone, Debug)]
#[pyclass]
pub struct Row {
    pub class: OrderedFloat<f64>,
    pub attributes: NamedValuesSetList,
}

impl fmt::Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {:?}", self.class, self.attributes)?;
        Ok(())
    }
}
