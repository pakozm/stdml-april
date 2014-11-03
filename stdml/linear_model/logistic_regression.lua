--[[
  Copyright (c) 2014 Francisco Zamora-Martinez (pakozm@gmail.com)

  The STDML is an extension for APRIL-ANN toolkit, and both are free software;
  you can redistribute it and/or modify it under the terms of the GNU General
  Public License version 3 as published by the Free Software Foundation
  
  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License along with
  this library; if not, write to the Free Software Foundation, Inc., 59 Temple
  Place, Suite 330, Boston, MA 02111-1307 USA
]]

-- imports
local class_util = require "stdml.classification_utils"

local check_target_classes = class_util.check_target_classes
local mop = matrix.op

----------------------------------------------------------------------------

local function validate_ds(self,va_data,l1,l2)
  local trainer = self.trainer
  local va_loss = trainer:validate_dataset(va_data)
  local coef = trainer:weights("coef_")
  if l1 then va_loss = va_loss + l1*mop.abs(coef):sum() end
  if l2 then va_loss = va_loss + 0.5*l2*coef:dot(coef) end
  return va_loss
end

local function train_ds(self,tr_data,va_data,l1,l2)
  local trainer = self.trainer
  local tr_loss = trainer:train_dataset(tr_data)
  local va_loss = validate_ds(self,va_data,l1,l2)
  return trainer,tr_loss,va_loss
end

----------------------------------------------------------------------------

local log_reg,log_reg_methods = class("logistic_regression")

april_set_doc(log_reg, { class = "class",
                         summary = "Logistic regression class" })

log_reg._NAME    = "stdml.linear_model.logistic_regression"
log_reg._VERSION = "0.1"

----------------------------------------------------------------------------

log_reg.constructor =
  april_doc{
    class = "method",
    summary = "Constructor",
    description = {
      "A logistic regression model has two important properties:",
      "self.coef_ and self.intercept_",
    },
    params = {
      l1 = { "L1 regularization parameter (by default 0.0) [optional]" },
      l2 = { "L2 regularization parameter (by default 0.0) [optional]" },
      fit_intercept = { "Boolean indicating if fit or not the intercept",
                        "(by default true) [optional]" },
      shuffle = { "A random number generator object (random() by default) [optional]" },
      tol = { "Tolerance value (by default 0.0001) [optional]", },
      max_epochs = { "Maximum number of epochs (by default 1000) [optional]", },
      min_epochs = { "Minimum number of epochs (by default 10) [optional]", },
      verbose = { "Boolean value, true for increase verbosity [optional]", },
      method = { "Which optimization method taken from ann.optimizer",
                 "(by default rprop) [optional]", },
      bunch_size = { "Bunch size (mini-batch size), by default it is the",
                     "min(num_samples, 1024) [optional]", },
    },
  } ..
  function(self,params)
    self.params = get_table_fields(
      { l1 = { type_match="number" },
        l2 = { type_match="number" },
        fit_intercept = { type_match="boolean", default=true },
        -- class_weight = { },
        shuffle = { isa_match=random, default=random() },
        tol = { type_match="number", default=0.0001 },
        max_epochs = { type_match="number", default=1000 },
        min_epochs = { type_match="number", default=10 },
        verbose = { type_match="boolean" },
        method = { type_match="string", default="rprop" },
        bunch_size = { type_match="number" },
      }, params)
    assert(ann.optimizer[self.params.method], "Needs a valid optimizer method")
  end

----------------------------------------------------------------------------

log_reg_methods.fit =
  april_doc{
    class = "method",
    summary = "Fits the model given features and targets",
    description = { "Training implements the 'pocket algorithm', keeping the",
                    "parameters which minimize validation loss.",
                    "In case validation is not given, training set will",
                    "be used as validation", },
    params = {
      "The training input features matrix",
      "The training target output matrix",
      "The validation input features matrix [optional]",
      "The validation target output matrix [optional]",
    },
    outputs = {
      "The validation loss, or training loss if not validation given",
      "The training loss (will be the same as 1st output if not validation given",
      "The number of iterations needed to reach the optimum value",
    },
  } ..
  function(self,x,y,val_x,val_y)
    local with_validation = val_x and val_y
    local function check_and_parse_x(x)
      assert(not class.is_a(x,matrix.sparse), "NOT IMPLEMENTED FOR SPARSE MATRICES")
      assert(class.is_a(x,matrix) or class.is_a(x,matrix.sparse),
             "Needs a matrix or matrix.sparse as input features")
      assert(#x:dim() == 2, "Needs a 2D matrix as input features")
      return x
    end
    local function check_and_parse_y(y)
      assert(class.is_a(y,matrix) or class.is_a(y,matrixInt32) or type(y)=="table",
             "Needs a matrix, matrixInt32 or table as output targets")
      if type(y)=="table" then y = matrix(y)
      elseif class.is_a(y,matrixInt32) then y = y:to_float()
      end
      local y_dim = y:dim()
      assert(#y_dim == 1 or (#y_dim==2 and y_dim[2]==1),
             "Needs a vector as output target")
      return y
    end
    local x     = check_and_parse_x(x)
    local y     = check_and_parse_y(y)
    assert(x:dim(1) == y:dim(1), "Needs same input/output number of samples")
    local val_x = check_and_parse_x(val_x or x)
    local val_y = check_and_parse_y(val_y or y)
    assert(val_x:dim(1) == val_y:dim(1),
           "Needs same input/output number of samples")
    local x_dim,y_dim = x:dim(),y:dim()
    local num_samples,num_features = x:dim(1),x:dim(2)
    local num_classes,class_dict = check_target_classes(y)
    self.num_classes = num_classes
    -- MODEL
    local num_outputs = (num_classes==2) and 1 or num_classes
    local model = ann.components.stack()
    model:push( ann.components.dot_product{ input=num_features, output=num_outputs,
                                            weights="coef_" } )
    if self.params.fit_intercept then
      model:push( ann.components.bias{ size=num_outputs, weights="intercept_" } )
    end
    model:push( ann.components.actf.log_logistic() )
    --
    local loss = (num_classes==2) and ann.loss.cross_entropy() or ann.loss.multi_class_cross_entropy()
    local bsize = self.params.bunch_size or math.min(1024, num_samples)
    local trainer = trainable.supervised_trainer(model, loss, bsize,
                                                 ann.optimizer[self.params.method]())
    trainer:build()
    if trainer:has_option("learning_rate") then
      trainer:set_option("learning_rate", 0.02)
    end
    if trainer:has_option("momentum") then
      trainer:set_option("momentum", 0.04)
    end
    if trainer:has_option("decay") then
      trainer:set_option("decay", 1e-05)
    end
    local coef_,intercept_ = trainer:weights("coef_"),trainer:weights("intercept_")
    self.coef_,self.intercept_= coef_,intercept_
    self.trainer = trainer
    coef_:zeros()
    if intercept_ then intercept_:zeros() end
    --
    local l1,l2 = self.params.l1,self.params.l2
    if l1 then trainer:set_layerwise_option("coef_", "L1_norm", l1) end
    if l2 then trainer:set_layerwise_option("coef_", "weight_decay", l2) end
    --
    local in_ds   = dataset.matrix(x)
    local out_ds  = dataset.matrix(y)
    local val_in_ds   = dataset.matrix(val_x)
    local val_out_ds  = dataset.matrix(val_y)
    if num_classes > 2 then
      out_ds = dataset.indexed(out_ds, { dataset.identity(num_classes) })
      val_out_ds = dataset.indexed(val_out_ds, { dataset.identity(num_classes) })
    end
    local tr_data = {
      input_dataset  = in_ds,
      output_dataset = out_ds,
      shuffle        = self.params.shuffle,
      replacement    = bsize,
    }
    local va_data = {
      input_dataset  = val_in_ds,
      output_dataset = val_out_ds,
    }
    local verbose = self.params.verbose
    local pocket = trainable.train_holdout_validation{
      stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(1.2),
      min_epochs = self.params.min_epochs,
      max_epochs = self.params.max_epochs,
      tolerance  = self.params.tol,
    }
    while pocket:execute(train_ds,self,tr_data,va_data,l1,l2) do
      if verbose then print(pocket:get_state_string()) end
    end
    local state  = pocket:get_state_table()
    --
    self.trainer = state.best
    self.coef_ = self.trainer:weights("coef_")
    self.intercept_ = self.trainer:weights("intercept_")
    --
    local tr_loss = validate_ds(self, { input_dataset = in_ds,
                                        output_dataset = out_ds }, l1, l2)
    local va_loss = with_validation and validate_ds(self, va_data, l1, l2) or tr_loss
    return va_loss, tr_loss, state.best_epoch
  end

----------------------------------------------------------------------------

function log_reg_methods:predict(x,th)
  local th  = th or 0.5
  local out = self.trainer:calculate(x)
  if self.num_classes > 2 then
    out:max(2)
  else
    out = out:exp():gt(th):to_float()
  end
  return out
end

----------------------------------------------------------------------------

function log_reg_methods:predict_log_proba(x)
  return self.trainer:calculate(x)
end

----------------------------------------------------------------------------

function log_reg_methods:predict_proba(x)
  return self.trainer:calculate(x):exp()
end

----------------------------------------------------------------------------

function log_reg_methods:transform(x,threshold)
  assert(self.coef_, "Needs a fitted model")
  local threshold = threshold or 0.0
  local gt_zero = matrix.op.abs(self.coef_):gt(threshold)
  local sel_cols = matrixBool(gt_zero:dim(2)):zeros()
  for i=1,gt_zero:dim(1) do
    sel_cols[{}] = sel_cols + gt_zero:select(1,i)
  end
  return x:index(2,sel_cols)
end

-----------------------------------------------------------------------------

local function two_class_test()
  local N=10000
  local val_N=1000
  local M=matrix
  local D=matrix.sparse.diag
  local rnd = random(1234)
  
  local function build_data(N)
    local x = matrix(N,2)
    local y = matrix(N,1)
    local N_2 = math.floor(N/2)
    -- first N_2 samples are drawn from mean (2,1) variance (2,2)
    stats.dist.normal(M{2,1},D{2,2}):sample(rnd, x( {1,N_2}, ':' ))
    -- last N_2 samples are drawn from mean (-2,-1) variance (0.3,0.3)
    stats.dist.normal(M{-2,-1},D{0.3,0.3}):sample(rnd, x( {N_2+1,N}, ':' ))
    -- first N_2 samples are of negative class
    y[{ {1,N_2} }] = 0
    -- first N_2 samples are of positive class
    y[{ {N_2,N} }] = 1
    return x,y
  end
  
  local x,y         = build_data(N)
  local val_x,val_y = build_data(val_N)
  local model = log_reg{ verbose = false, l2 = 0.01, shuffle = random(1234) }
  local va_loss,tr_loss,n = model:fit(x,y,val_x,val_y)
  
  print(model.coef_)
  print(model.intercept_)
  --print(model:predict(x))
  --print(model:predict_proba(x))
  print(model:predict(val_x):sum())
  local val_out = model:predict_proba(val_x)
  
  local auc = metrics.roc(val_out,val_y):compute_area()
  print("LOSS:", va_loss, tr_loss, n)
  print("AUC:", auc)
  
  matrix.join(2, val_x, val_out, val_y):toTabFilename("JARL")
end

function log_reg.test()
  two_class_test()
end

-----------------------------------------------------------------------------

return log_reg
