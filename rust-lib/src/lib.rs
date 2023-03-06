// from https://github.com/ZakSingh/omelette
use egg::*;
use once_cell::sync::Lazy;

// use log::{debug, error, info, log_enabled, Level};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::{borrow::Cow, fmt::Display, hash::Hash, time::Duration};

use pyo3::AsPyPointer;
use pyo3::{
    basic::CompareOp,
    prelude::*,
    types::{PyList, PyString, PyTuple, PyType},
};

fn py_eq(a: &PyAny, b: impl ToPyObject) -> bool {
    a.rich_compare(b, CompareOp::Eq)
        .expect("Failed to compare")
        .is_true()
        .expect("Failed to extract bool")
}

macro_rules! py_object {
    (impl $t:ty { $($rest:tt)* }) => {
        #[pymethods]
        impl $t {
            $($rest)*

            fn __str__(&self) -> String {
                self.0.to_string()
            }

            fn __repr__(&self) -> String {
                format!(concat!(stringify!($t), "({})"), self.0)
            }

            fn __richcmp__(&self, other: Self, op: CompareOp) -> bool {
                match op {
                    CompareOp::Lt => self.0 < other.0,
                    CompareOp::Le => self.0 <= other.0,
                    CompareOp::Eq => self.0 == other.0,
                    CompareOp::Ne => self.0 != other.0,
                    CompareOp::Gt => self.0 > other.0,
                    CompareOp::Ge => self.0 >= other.0,
                }
            }
        }
    };
}

#[pyclass]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Id(egg::Id);

py_object!(impl Id {});

#[pyclass]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Var(egg::Var);

py_object!(impl Var {
    #[new]
    fn new(str: &PyString) -> Self {
        Self::from_str(str.to_string_lossy().as_ref())
    }
});

impl Var {
    fn from_str(str: &str) -> Self {
        let v = format!("?{}", str);
        Var(v.parse().unwrap())
    }
}

#[derive(Debug, Clone)]
struct PyLang {
    obj: PyObject,
    children: Vec<egg::Id>,
}

impl PyLang {
    fn op(ty: &PyType, children: impl IntoIterator<Item = egg::Id>) -> Self {
        let any = ty.as_ref();
        let py = any.py();
        Self {
            obj: any.to_object(py),
            children: children.into_iter().collect(),
        }
    }

    fn leaf(any: &PyAny) -> Self {
        struct Hashable {
            obj: PyObject,
            hash: isize,
        }

        impl Hash for Hashable {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.hash.hash(state);
            }
        }

        impl PartialEq for Hashable {
            fn eq(&self, other: &Self) -> bool {
                let py = unsafe { Python::assume_gil_acquired() };
                py_eq(self.obj.as_ref(py), &other.obj)
            }
        }

        impl Eq for Hashable {}

        static LEAVES: Lazy<Mutex<hashbrown::HashSet<Hashable>>> = Lazy::new(Default::default);

        let hash = any.hash().expect("failed to hash");
        let py = any.py();
        let obj = any.to_object(py);

        let mut leaves = LEAVES.lock().unwrap();
        let hashable = leaves.get_or_insert(Hashable { obj, hash });

        Self {
            obj: hashable.obj.clone(),
            children: vec![],
        }
    }

    fn to_object<T: IntoPy<PyObject>>(&self, py: Python, f: impl FnMut(egg::Id) -> T) -> PyObject {
        if self.is_leaf() {
            self.obj.clone()
        } else {
            let children = self.children.iter().copied().map(f);
            let args = PyTuple::new(py, children.map(|o| o.into_py(py)));
            self.obj.call1(py, args).expect("Failed to construct")
        }
    }
}

impl PartialEq for PyLang {
    fn eq(&self, other: &Self) -> bool {
        self.obj.as_ptr() == other.obj.as_ptr() && self.children == other.children
    }
}

impl Hash for PyLang {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.obj.as_ptr().hash(state);
        self.children.hash(state);
    }
}

impl Ord for PyLang {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).expect("comparison failed")
    }
}

impl PartialOrd for PyLang {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.obj.as_ptr().partial_cmp(&other.obj.as_ptr()) {
            Some(Ordering::Equal) => {}
            ord => return ord,
        }
        self.children.partial_cmp(&other.children)
    }
}

impl Eq for PyLang {}

impl egg::Language for PyLang {
    fn matches(&self, other: &Self) -> bool {
        self.obj.as_ptr() == other.obj.as_ptr() && self.children.len() == other.children.len()
    }

    fn children(&self) -> &[egg::Id] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [egg::Id] {
        &mut self.children
    }
}

impl Display for PyLang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Python::with_gil(|py| match self.obj.as_ref(py).str() {
            Ok(s) => s.fmt(f),
            Err(_) => "<<NODE>>".fmt(f),
        })
    }
}

#[pyclass]
struct Pattern {
    pattern: egg::Pattern<PyLang>,
}

#[pymethods]
impl Pattern {
    #[new]
    fn new(tree: &PyAny) -> Self {
        let mut ast = egg::PatternAst::default();
        build_pattern(&mut ast, tree);
        let pattern = egg::Pattern::from(ast);
        Self { pattern }
    }
}

fn build_pattern(ast: &mut egg::PatternAst<PyLang>, tree: &PyAny) -> egg::Id {
    if let Ok(id) = tree.extract::<Id>() {
        panic!("Ids are unsupported in patterns: {}", id.0)
    } else if let Ok(var) = tree.extract::<Var>() {
        ast.add(egg::ENodeOrVar::Var(var.0))
    } else if let Ok(tuple) = tree.downcast::<PyTuple>() {
        let op = PyLang::op(
            tree.get_type(),
            tuple.iter().map(|child| build_pattern(ast, child)),
        );
        ast.add(egg::ENodeOrVar::ENode(op))
    } else {
        ast.add(egg::ENodeOrVar::ENode(PyLang::leaf(tree)))
    }
}

#[pyclass]
struct Rewrite {
    rewrite: egg::Rewrite<PyLang, PyAnalysis>,
}

#[pymethods]
impl Rewrite {
    #[new]
    #[args(name = "\"\"")]
    fn new(lhs: &PyAny, rhs: &PyAny, name: &str) -> Self {
        let searcher = Pattern::new(lhs).pattern;
        let applier = Pattern::new(rhs).pattern;
        let mut name = Cow::Borrowed(name);
        if name == "" {
            name = Cow::Owned(format!("{} => {}", searcher, applier));
        }
        let name_str: &str = name.borrow();
        let rewrite =
            egg::Rewrite::new(name_str, searcher, applier).expect("Failed to create rewrite");
        Rewrite { rewrite }
    }

    #[getter]
    fn name(&self) -> &str {
        self.rewrite.name.as_str()
    }
}

#[derive(Default, Clone)]
struct PyAnalysis {
    eval: Option<PyObject>,
}

impl egg::Analysis<PyLang> for PyAnalysis {
    type Data = Option<PyObject>;

    fn make(egraph: &egg::EGraph<PyLang, Self>, enode: &PyLang) -> Self::Data {
        let eval = egraph.analysis.eval.as_ref()?;
        let py = unsafe { Python::assume_gil_acquired() };

        // collect the children if they are not `None` in python
        let children: Vec<&PyAny> = Vec::with_capacity(enode.len());
        for &id in enode.children() {
            let any = egraph[id].data.as_ref()?.as_ref(py);
            if any.is_none() {
                return None;
            } else {
                // FIXME: Is this needed? What does it do?
                // children.push(any)
            }
        }

        let res = eval
            .call1(py, (enode.obj.clone(), children))
            .expect("Failed to call eval");
        if res.is_none(py) {
            None
        } else {
            Some(res)
        }
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> egg::DidMerge {
        let py = unsafe { Python::assume_gil_acquired() };
        let aa = a.as_ref().map(|obj| obj.as_ref(py)).filter(|r| r.is_none());
        let bb = b.as_ref().map(|obj| obj.as_ref(py)).filter(|r| r.is_none());
        match (aa, bb) {
            (None, None) => egg::DidMerge(false, false),
            (None, Some(bb)) => {
                *a = Some(bb.to_object(py));
                egg::DidMerge(true, false)
            }
            (Some(_), None) => egg::DidMerge(false, true),
            (Some(aa), Some(bb)) => {
                if !py_eq(aa, bb) {
                    panic!("Failed to merge")
                }
                egg::DidMerge(false, false)
            }
        }
    }

    fn modify(egraph: &mut egg::EGraph<PyLang, Self>, id: egg::Id) {
        let obj = egraph[id].data.clone();
        if let Some(obj) = obj {
            let py = unsafe { Python::assume_gil_acquired() };
            let id2 = add_rec(egraph, obj.as_ref(py));
            egraph.union(id, id2);
        }
    }
}

#[pyclass]
struct EGraph {
    egraph: egg::EGraph<PyLang, PyAnalysis>,
}

type Runner = egg::Runner<PyLang, PyAnalysis, ()>;

#[pymethods]
impl EGraph {
    #[new]
    fn new(eval: Option<PyObject>) -> Self {
        Self {
            egraph: egg::EGraph::new(PyAnalysis { eval }),
        }
    }

    fn add(&mut self, expr: &PyAny) -> Id {
        Id(add_rec(&mut self.egraph, expr))
    }

    #[args(exprs = "*")]
    fn union(&mut self, exprs: &PyTuple) -> bool {
        assert!(exprs.len() > 1);
        let mut exprs = exprs.iter();
        let id = self.add(exprs.next().unwrap()).0;
        let mut did_something = false;
        for expr in exprs {
            let added = self.add(expr);
            did_something |= self.egraph.union(id, added.0);
        }
        did_something
    }

    #[args(exprs = "*")]
    fn equiv(&mut self, exprs: &PyTuple) -> bool {
        assert!(exprs.len() > 1);
        let mut exprs = exprs.iter();
        let id = self.add(exprs.next().unwrap()).0;
        let mut all_equiv = true;
        for expr in exprs {
            let added = self.add(expr);
            all_equiv &= added.0 == id
        }
        all_equiv
    }

    fn rebuild(&mut self) -> usize {
        self.egraph.rebuild()
    }

    fn clone(&mut self) -> EGraph {
        Self {
            egraph: self.egraph.clone(),
        }
    }

    fn graphviz(&mut self, path: &str) {
        self.egraph.dot().to_png(path).unwrap()
    }

    #[args(
        iter_limit = "10",
        time_limit = "10.0",
        node_limit = "100_000",
        use_backoff = "false"
    )]
    fn run(
        &mut self,
        py: Python,
        rewrites: &PyList,
        iter_limit: usize,
        time_limit: f64,
        node_limit: usize,
        use_backoff: bool
    ) -> PyResult<Vec<PyObject>> {
        let refs = rewrites
            .iter()
            .map(FromPyObject::extract)
            .collect::<PyResult<Vec<PyRef<Rewrite>>>>()?;
        let egraph = std::mem::take(&mut self.egraph);
        let scheduled_runner = Runner::default();

        let runner = if use_backoff {
            scheduled_runner
                .with_iter_limit(iter_limit)
                .with_node_limit(node_limit)
                .with_time_limit(Duration::from_secs_f64(time_limit))
                .with_egraph(egraph)
                .run(refs.iter().map(|r| &r.rewrite))
        } else {
            scheduled_runner
                .with_iter_limit(iter_limit)
                .with_node_limit(node_limit)
                .with_time_limit(Duration::from_secs_f64(time_limit))
                .with_egraph(egraph)
                .with_scheduler(SimpleScheduler)
                .run(refs.iter().map(|r| &r.rewrite))
        };

        let num_applications: usize = runner
            .iterations
            .iter()
            .map(|i| i.applied.values().sum::<usize>())
            .sum();

        let egraph_nodes: usize = runner.egraph.total_size();
        let egraph_classes: usize = runner.egraph.number_of_classes();
        self.egraph = runner.egraph;

        match runner.stop_reason.unwrap() {
            StopReason::IterationLimit(_) => Ok(vec![
                "ITERATION_LIMIT".to_object(py),
                num_applications.to_object(py),
                egraph_nodes.to_object(py),
                egraph_classes.to_object(py),
            ]),
            StopReason::NodeLimit(_) => Ok(vec![
                "NODE_LIMIT".to_object(py),
                num_applications.to_object(py),
                egraph_nodes.to_object(py),
                egraph_classes.to_object(py),
            ]),
            StopReason::Saturated => Ok(vec![
                "SATURATED".to_object(py),
                num_applications.to_object(py),
                egraph_nodes.to_object(py),
                egraph_classes.to_object(py),
            ]),
            StopReason::TimeLimit(_) => {
                if egraph_nodes > node_limit {
                    Ok(vec![
                        "NODE_LIMIT".to_object(py),
                        num_applications.to_object(py),
                        egraph_nodes.to_object(py),
                        egraph_classes.to_object(py),
                    ])
                } else {
                    Ok(vec![
                        "TIME_LIMIT".to_object(py),
                        num_applications.to_object(py),
                        egraph_nodes.to_object(py),
                        egraph_classes.to_object(py),
                    ])
                }
            }
            StopReason::Other(_) => Ok(vec![
                "OTHER".to_object(py),
                num_applications.to_object(py),
                egraph_nodes.to_object(py),
                egraph_classes.to_object(py),
            ]),
        }
    }

    #[args(exprs = "*")]
    fn extract(&mut self, py: Python, exprs: &PyTuple) -> SingletonOrTuple<(usize, PyObject)> {
        self.egraph.rebuild();
        let ids: Vec<egg::Id> = exprs.iter().map(|expr| self.add(expr).0).collect();
        assert_eq!(ids.len(), 1);

        // this will compute the cost
        let extractor = egg::Extractor::new(&self.egraph, egg::AstSize);

        ids.iter()
            .map(|&id| {

                // greedy extraction
                let (cost, recexpr) = extractor.find_best(id);
                (cost, reconstruct(py, &recexpr))
            })
            .collect()
    }

    #[args(exprs = "*")]
    fn extraction_entry_point(&mut self, exprs: &PyTuple) -> Vec<String> {
        self.egraph.rebuild();
        let ids: Vec<egg::Id> = exprs.iter().map(|expr| self.add(expr).0).collect();
        assert_eq!(ids.len(), 1);
        ids.iter().map(|id| id.to_string()).collect()
    }

    fn classes(&mut self, py: Python) -> PyObject {
        let eg = &self.egraph;
        let mut ids: Vec<egg::Id> = eg.classes().map(|c| c.id).collect();
        ids.sort();

        let mut nodes_by_class: HashMap<String, (Option<PyObject>, Vec<PyObject>)> = HashMap::new();
        // Collect all the nodes, grouped by class ID
        for id in ids {
            let data = eg[id].data.clone();
            let mut class_nodes: Vec<PyObject> = vec![];
            for node in &eg[id].nodes {
                class_nodes.push(node.to_object(py, |id| Id(id)));
            }
            nodes_by_class.insert(id.to_string(), (data, class_nodes));
        }

        nodes_by_class.to_object(py)
    }

    // return a list of what each Eclass point to (other Eclasses)
    fn eclass2class_connection(&mut self, py: Python) -> PyObject {
        println!("eclass2class_connection deprecated");
        let eg = &self.egraph;

        // all Eclass Id
        let mut ids: Vec<egg::Id> = eg.classes().map(|c| c.id).collect();
        ids.sort();

        // Eclass ID -> its point-to Eclass ID
        let mut edges: HashMap<String, Vec<String>> = HashMap::new();
        // for each Eclass
        for id in ids {
            let mut vec = Vec::new();
            let mut set = HashSet::new();
            // for each Enodes In this Eclass
            for enode in &eg[id].nodes {
                for &child in enode.children() {
                    let s = child.to_string();
                    if !set.contains(&s) {
                        set.insert(s);
                        vec.push(child.to_string());
                    }
                }
            }
            edges.insert(id.to_string(), vec);
        }
        edges.to_object(py)
    }

    fn num_eclasses(&mut self) -> usize {
        *&self.egraph.number_of_classes()
    }

    fn num_enodes(&mut self) -> usize {
        *&self.egraph.total_number_of_nodes()
    }

    fn total_size(&mut self) -> usize {
        *&self.egraph.total_size()
    }

    fn eclass_ids(&mut self) -> Vec<String> {
        let eg = &self.egraph;
        let mut ids: Vec<String> = eg.classes().map(|c| c.id.to_string()).collect();
        ids.sort();
        return ids;
    }

    fn match_rules(&mut self, py: Python, rewrites: &PyList) -> PyObject {
        // Search for each rewrite rule. Return a dict of "rule_name": [eclass_ids] for each
        let refs = rewrites
            .iter()
            .map(FromPyObject::extract)
            .collect::<PyResult<Vec<PyRef<Rewrite>>>>()
            .unwrap();

        let matches: HashMap<&str, Vec<String>> = refs
            .iter()
            .map(|r| &r.rewrite)
            .map(|rw| {
                (
                    rw.name.as_str(),
                    rw.search(&self.egraph)
                        .into_iter()
                        .map(|m| m.eclass.to_string())
                        .collect(),
                )
            })
            .collect();

        matches.to_object(py)
    }
}

fn reconstruct(py: Python, recexpr: &RecExpr<PyLang>) -> PyObject {
    let mut objs = Vec::<PyObject>::with_capacity(recexpr.as_ref().len());
    for node in recexpr.as_ref() {
        let obj = node.to_object(py, |id| objs[usize::from(id)].clone());
        objs.push(obj)
    }
    objs.pop().unwrap()
}

fn add_rec(egraph: &mut egg::EGraph<PyLang, PyAnalysis>, expr: &PyAny) -> egg::Id {
    if let Ok(Id(id)) = expr.extract() {
        egraph.find(id)
    } else if let Ok(Var(var)) = expr.extract() {
        panic!("Can't add a var: {}", var)
    } else if let Ok(tuple) = expr.downcast::<PyTuple>() {
        let enode = PyLang::op(
            expr.get_type(),
            tuple.iter().map(|child| add_rec(egraph, child)),
        );
        egraph.add(enode)
    } else {
        egraph.add(PyLang::leaf(expr))
    }
}

struct SingletonOrTuple<T>(Vec<T>);

impl<T: IntoPy<PyObject>> IntoPy<PyObject> for SingletonOrTuple<T> {
    fn into_py(mut self, py: Python) -> PyObject {
        match self.0.len() {
            0 => panic!("Shouldn't be empty"),
            1 => self.0.pop().unwrap().into_py(py),
            _ => PyTuple::new(py, self.0.into_iter().map(|x| x.into_py(py))).into_py(py),
        }
    }
}

impl<T: IntoPy<PyObject>> FromIterator<T> for SingletonOrTuple<T> {
    fn from_iter<TS: IntoIterator<Item = T>>(iter: TS) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
// fn rejoice(_py: Python, m: &PyModule) -> PyResult<()> {
fn rust_lib(_py: Python, m: &PyModule) -> PyResult<()> {
    // env_logger::init();
    // pyo3_log::init();

    m.add_class::<EGraph>()?;
    m.add_class::<Id>()?;
    m.add_class::<Var>()?;
    m.add_class::<Pattern>()?;
    m.add_class::<Rewrite>()?;

    #[pyfn(m)]
    fn vars(vars: &PyString) -> SingletonOrTuple<Var> {
        let s = vars.to_string_lossy();
        s.split_whitespace().map(|s| Var::from_str(s)).collect()
    }
    Ok(())
}
