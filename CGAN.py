class UDA:
    def __init__(self, n_classes=19, x_data=None, y_data=None,
                 x_target=None, val_data=None, val_labels=None,
                 batch_size=5, shuffle=True, cmap=None, source_size=None, target_size=None,
                 learning_rate=0.00025, epochs=32, reference_path=None,
                 dim=(None, None), name="UDA", **kwargs):
        self.source_size = source_size
        self.batch_size = batch_size
        self.total_steps = int(250000 // batch_size)
        self.input_size = source_size
        self.input_size_target = target_size
        self.base_learning_rate_G = 0.00025
        self.learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(self.base_learning_rate_G,
                                                                           self.total_steps,
                                                                           0.0000000013862846,
                                                                           power=0.9)

        self.momentum = 0.9
        self.n_classes = n_classes
        self.random_seed = 888
        self.epochs = epochs

        self.base_learning_rate_D = 0.0001
        self.learning_rate_D1 = tf.keras.optimizers.schedules.PolynomialDecay(self.base_learning_rate_D,
                                                                              self.total_steps,
                                                                              0.0000000034657,
                                                                              power=0.9)

        self.learning_rate_D2 = tf.keras.optimizers.schedules.PolynomialDecay(self.base_learning_rate_D,
                                                                              self.total_steps,
                                                                              0.0000000034657,
                                                                              power=0.9)
        self.lambda_seg = 0.1
        self.labmda_adv_target1 = 0.00025
        self.labmda_adv_target2 = 0.002

        # self.model = resnet_101(n_classes=n_classes)
        self.model = DeeplabV3Plus(num_classes=n_classes)
        self.D1 = FCDiscriminator(num_classes=n_classes)
        self.D2 = FCDiscriminator(num_classes=n_classes)

        self.classifier_loss_function = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.bce_loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.gen_optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate,
                                                     momentum=self.momentum)

        self.disc_optimizer1 = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_D1, beta_1=0.9, beta_2=0.99)
        self.disc_optimizer2 = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_D2, beta_1=0.9, beta_2=0.99)

        self.data_generator = DataGenerator(source_data=x_data, source_labels=y_data,
                                            batch_size=batch_size, resize_size=source_size, target=False, val=False)

        self.target_generator = DataGenerator(source_data=x_target, source_labels=None,
                                            batch_size=batch_size, resize_size=target_size, target=True, val=False)

        self.validation_data_generator = DataGenerator(source_data=val_data, source_labels=val_labels,
                                            batch_size=batch_size, resize_size=target_size, target=False, val=True)

        self.train_accuracy1 = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_accuracy2 = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_disc_on_source_loss = tf.keras.metrics.Mean(name='train_disc_loss')
        self.train_disc_loss = tf.keras.metrics.Mean(name='train_disc_loss')
        self.train_gen_loss = tf.keras.metrics.Mean(name='train_gen_loss')

        self.validation_metrics = tf.keras.metrics.SparseCategoricalAccuracy()
        self.validation_iou_metrics = tf.keras.metrics.MeanIoU(num_classes=n_classes)
        self.validation_iou_gen_metrics = tf.keras.metrics.MeanIoU(num_classes=n_classes)
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.test_gen_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.in_train_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.in_train_iou = 0.0

        self.disc_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.disc_accuracy_true = tf.keras.metrics.BinaryAccuracy()
        self.disc_accuracy_fake = tf.keras.metrics.BinaryAccuracy()
        self.gen_accuracy = tf.keras.metrics.BinaryAccuracy()

        self.train_summary_writer = tf.summary.create_file_writer("./logs/fit")

        self.power = 0.9
        self.acc_level = 0.0
        self.on_init()

    def on_init(self):
        dummy_tensor = np.random.uniform(size=(self.batch_size, *self.source_size, 3))
        y = self.model(dummy_tensor)


    def draw_labels(self, sample, true_labels, preds, batch_size, epoch=0, step=0, tr=False, gen=False):
        pred_mask = tf.argmax(preds, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]

        prm = np.asarray(pred_mask)
        pred_mask = prm.copy()

        for b in range(self.batch_size):
            pred_mask[b][0, 0, 0] = 255

        pred_mask = tf.where(pred_mask == 255, self.n_classes + 1, pred_mask)
        true_labels = tf.where(true_labels == 255, self.n_classes + 1, true_labels)

        title = ['Input Image', 'True Labels', 'Predicted Mask']

        if batch_size == 1:
            fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15, 15))
            ax1.set_title(title[0])
            ax1.imshow(tf.keras.preprocessing.image.array_to_img(sample[0]))
            ax1.axis('off')

            ax2.set_title(title[1])
            ax2.imshow(tf.keras.preprocessing.image.array_to_img(true_labels[0]))
            ax2.axis('off')

            ax3.set_title(title[2])
            ax3.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[0]))
            ax3.axis('off')
            if tr:
                plt.savefig(f'UDA_results_on_{epoch + 1}_epoch_at_step_{step}_TRAINING_Full_DS.jpg')
            else:
                if gen:
                    plt.savefig(f'UDA_results_after_{epoch + 1}_epoch_VALIDATION_Full_DS_With_Gen.jpg')
                else:
                    plt.savefig(f'UDA_results_after_{epoch + 1}_epoch_VALIDATION_Full_DS.jpg')
        else:
            fig, ax = plt.subplots(batch_size, 3, figsize=(15, 15))

            for i in range(batch_size):
                display_list = [sample[i], true_labels[i], pred_mask[i]]
                for j in range(3):
                    ax[i, j].set_title(title[j])
                    ax[i, j].imshow(tf.keras.preprocessing.image.array_to_img(display_list[j]))
                    ax[i, j].axis('off')
                if tr:
                    plt.savefig(f'UDA_results_on_{epoch + 1}_epoch_at_step_{step}_TRAINING_Full_DS.jpg')
                else:
                    if gen:
                        plt.savefig(f'UDA_results_after_{epoch + 1}_epoch_VALIDATION_Full_DS_With_Gen.jpg')
                    else:
                        plt.savefig(f'UDA_results_after_{epoch + 1}_epoch_VALIDATION_Full_DS.jpg')
        plt.close(fig)

    def fast_hist(self, a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(self, hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_batch_mIoU(self, labels, predicted):
        pred_mask = tf.argmax(predicted, axis=-1)
        predicted = pred_mask[..., tf.newaxis]

        labels = np.asarray(labels, np.int8)
        predicted = np.asarray(predicted, np.int8)

        assert labels.shape == predicted.shape, 'Labels and logits shapes are different'

        """
        Compute IoU given the predicted colorized images and
        """
        # print('Num classes', self.n_classes)
        batch_hist = np.zeros((self.n_classes, self.n_classes), np.int32)
        for i in range(self.batch_size):
            batch_hist += self.fast_hist(labels[i, :, :, :].flatten(), predicted[i, :, :, :].flatten(), self.n_classes)

        return batch_hist

    def compute_mIoU(self, total_hist):
        per_class_mIoU = self.per_class_iu(total_hist)
        return round(np.nanmean(per_class_mIoU) * 100, 2)

    def compute_train_acc(self, predict, label, ignore_label=255):

        label_mask = (label >= 0) * (label != ignore_label)
        label_mask = tf.constant(label_mask, tf.float32)

        label = label_mask * label
        predict = predict * label_mask

        self.train_accuracy2.update_state(label, predict)

    def compute_test_acc(self, predict, label, ignore_label=255):
        label_mask = (label >= 0) * (label != ignore_label)
        label_mask = tf.constant(label_mask, tf.float32)

        label = label_mask * label
        predict = predict * label_mask

        self.in_train_test_accuracy.update_state(label, predict)

    def compute_val_acc(self, predict, label, ignore_label=255):
        label_mask = (label >= 0) * (label != ignore_label)
        label_mask = tf.constant(label_mask, tf.float32)

        label = label_mask * label
        predict = predict * label_mask

        self.test_gen_accuracy.update_state(label, predict)

    def Crossentropy(self, label, predict, ignore_label=255):

        label_mask = (label >= 0) * (label != ignore_label)
        label_mask = tf.constant(label_mask, tf.float32)

        label = label * label_mask
        predict = predict * label_mask

        loss = self.classifier_loss_function(label, predict)

        return loss

    def __train_step(self, source_batch, labels_batch, target_batch, epoch, step):
        self.train_gen_loss.reset_states()
        self.train_disc_loss.reset_states()

        loss_seg_value1 = 0
        loss_seg_value2 = 0
        loss_adv_target_value1 = 0
        loss_adv_target_value2 = 0
        loss_D_value1 = 0
        loss_D_value2 = 0
        accum_gen_gradient = [tf.zeros_like(this_var) for this_var in self.model.trainable_variables]
        accum_disc1_gradient = [tf.zeros_like(this_var) for this_var in self.D1.trainable_variables]
        accum_disc2_gradient = [tf.zeros_like(this_var) for this_var in self.D2.trainable_variables]

        # train G:
        # train with source:
        with tf.GradientTape() as gen1_tape:
            p1, p2 = self.model(source_batch, training=True)

            loss_seg1 = self.Crossentropy(labels_batch, p1)
            loss_seg2 = self.Crossentropy(labels_batch, p2)
            loss = loss_seg2 + self.lambda_seg * loss_seg1

        gen_segm_grad = gen1_tape.gradient(loss, self.model.trainable_variables)
        loss_seg_value1 += loss_seg1
        loss_seg_value2 += loss_seg2
        accum_gen_gradient = [(acum_grad + grad) for acum_grad, grad in zip(accum_gen_gradient, gen_segm_grad)]

        # train G:
        # train with target:
        with tf.GradientTape() as gen2_tape:
            pt1, pt2 = self.model(target_batch, training=True)
            D_out1 = self.D1(tf.nn.softmax(pt1))
            D_out2 = self.D2(tf.nn.softmax(pt2))
            loss_adv_target1 = self.bce_loss_function(tf.zeros_like(D_out1), D_out1)
            loss_adv_target2 = self.bce_loss_function(tf.zeros_like(D_out2), D_out2)
            loss = self.labmda_adv_target1 * loss_adv_target1 + self.labmda_adv_target2 * loss_adv_target2

        gen_adv_grad = gen2_tape.gradient(loss, self.model.trainable_variables)
        loss_adv_target_value1 += loss_adv_target1
        loss_adv_target_value2 += loss_adv_target2
        accum_gen_gradient = [(acum_grad + grad) for acum_grad, grad in zip(accum_gen_gradient, gen_adv_grad)]

        # train D:
        # train with source:
        with tf.GradientTape() as disc1_tape, \
                tf.GradientTape() as disc2_tape:
            p1, p2 = self.model(source_batch, training=True)

            D_out11 = self.D1(tf.nn.softmax(p1))
            D_out21 = self.D2(tf.nn.softmax(p2))

            loss_D1 = self.bce_loss_function(tf.zeros_like(D_out11), D_out11)
            loss_D2 = self.bce_loss_function(tf.zeros_like(D_out21), D_out21)

            loss_D1 /= 2
            loss_D2 /= 2

        disc1_src_grad = disc1_tape.gradient(loss_D1, self.D1.trainable_variables)
        disc2_src_grad = disc2_tape.gradient(loss_D2, self.D2.trainable_variables)

        accum_disc1_gradient = [(acum_grad + grad) for acum_grad, grad in zip(accum_disc1_gradient, disc1_src_grad)]
        accum_disc2_gradient = [(acum_grad + grad) for acum_grad, grad in zip(accum_disc2_gradient, disc2_src_grad)]
        loss_D_value1 += loss_D1
        loss_D_value2 += loss_D2

        # train with target:
        with tf.GradientTape() as disc1_tape, \
                tf.GradientTape() as disc2_tape:
            pt1, pt2 = self.model(target_batch, training=True)

            D_out12 = self.D1(tf.nn.softmax(pt1))
            D_out22 = self.D2(tf.nn.softmax(pt2))

            loss_D1 = self.bce_loss_function(tf.ones_like(D_out12), D_out12)
            loss_D2 = self.bce_loss_function(tf.ones_like(D_out22), D_out22)

            loss_D1 /= 2
            loss_D2 /= 2

        disc1_trg_grad = disc1_tape.gradient(loss_D1, self.D1.trainable_variables)
        disc2_trg_grad = disc2_tape.gradient(loss_D2, self.D2.trainable_variables)

        loss_D_value1 += loss_D1
        loss_D_value2 += loss_D2

        accum_disc1_gradient = [(acum_grad + grad) for acum_grad, grad in zip(accum_disc1_gradient, disc1_trg_grad)]
        accum_disc2_gradient = [(acum_grad + grad) for acum_grad, grad in zip(accum_disc2_gradient, disc2_trg_grad)]
        #
        # accum_gen_gradient = [this_grad / len(self.data_generator) for this_grad in accum_gen_gradient]
        # accum_disc1_gradient = [this_grad / len(self.data_generator) for this_grad in accum_disc1_gradient]
        # accum_disc2_gradient = [this_grad / len(self.data_generator) for this_grad in accum_disc2_gradient]

        self.gen_optimizer.apply_gradients(zip(accum_gen_gradient, self.model.trainable_variables))
        self.disc_optimizer1.apply_gradients(zip(accum_disc1_gradient, self.D1.trainable_variables))
        self.disc_optimizer2.apply_gradients(zip(accum_disc2_gradient, self.D2.trainable_variables))

        self.compute_train_acc(p2, labels_batch)

        if step % 500 == 0:
            self.draw_labels(source_batch, labels_batch, p2, self.batch_size, epoch, step, tr=True)

        if step % 15 == 0:
            x_batch, y_batch = self.validation_data_generator.get_sample_batch()
            _, val_p = self.model(x_batch, training=False)
            self.compute_test_acc(val_p, y_batch)

            batch_hist = self.compute_batch_mIoU(y_batch, val_p)
            self.in_train_iou = self.compute_mIoU(batch_hist)

        with self.train_summary_writer.as_default():
            tf.summary.scalar('segmentation_loss1', loss_seg_value1, step=len(self.data_generator) * epoch + step)
            tf.summary.scalar('segmentation_loss2', loss_seg_value2, step=len(self.data_generator) * epoch + step)
            tf.summary.scalar('gen_adv_loss1', loss_adv_target_value1, step=len(self.data_generator) * epoch + step)
            tf.summary.scalar('gen_adv_loss2', loss_adv_target_value2, step=len(self.data_generator) * epoch + step)
            tf.summary.scalar('D1_loss', loss_D_value1, step=len(self.data_generator) * epoch + step)
            tf.summary.scalar('D2_loss', loss_D_value2, step=len(self.data_generator) * epoch + step)

            # tf.summary.scalar('pred1_acc', self.train_accuracy1.result(), step=len(self.data_generator)*epoch+step)
            tf.summary.scalar('pred2_acc', self.train_accuracy2.result(), step=len(self.data_generator) * epoch + step)
            tf.summary.scalar('in_train_test_acc', self.in_train_test_accuracy.result(),
                              step=len(self.data_generator) * epoch + step)
            tf.summary.scalar('in_train_mIoU', self.in_train_iou, step=len(self.data_generator) * epoch + step)

        # self.train_accuracy1.reset_states()
        self.train_accuracy2.reset_states()

    def __test_step(self, source_batch):
        # prediction, _ = self.VGG_encoder_decoder(source_batch, training=False, use_generator=False)

        # prediction_gen, _ = self.VGG_encoder_decoder(source_batch, training=False, use_generator=True)
        _, p2 = self.model(source_batch, training=False)
        # pred1 = tf.image.resize(pred1, [self.input_size_target[0], self.input_size_target[1]], method='bilinear')
        # pred2 = self.deconv(p2, training=False)
        # pred2 = tf.image.resize(p2, [self.input_size_target[0], self.input_size_target[1]], method='bilinear')
        pred2 = p2
        return _, tf.nn.softmax(pred2)

    def train(self):
        metrics_names = ['gen_loss', 'disc_loss', 'train_accuracy']
        for epoch in range(self.epochs):
            prog_bar = Progbar(self.data_generator.get_num_samples(), stateful_metrics=metrics_names)
            print("\nepoch {}/{}".format(epoch + 1, self.epochs))
            for i in range(len(self.data_generator)):
                source_batch, labels_batch = self.data_generator[i]
                target_batch = self.target_generator[i]
                self.__train_step(source_batch, labels_batch, target_batch, epoch, i)

                values = [('gen_loss', self.train_gen_loss.result()), ('disc_loss', self.train_disc_loss.result())]

                prog_bar.update(i * self.batch_size, values=values)

            # Run a validation loop at the end of each epoch.

            hist = np.zeros((self.n_classes, self.n_classes))
            for i in range(len(self.validation_data_generator)):
                x_batch_val, y_batch_val = self.validation_data_generator[i]
                _, val_logits_2 = self.__test_step(x_batch_val)
                #
                # pred_mask = tf.argmax(val_logits, axis=-1)
                # pred_mask = pred_mask[..., tf.newaxis]
                #
                # self.validation_iou_metrics.update_state(y_batch_val, pred_mask)
                # self.test_accuracy.update_state(y_batch_val, val_logits)

                # pred_mask = tf.argmax(val_logits_2, axis=-1)
                # pred_mask = pred_mask[..., tf.newaxis]

                # self.validation_iou_gen_metrics.update_state(y_batch_val, pred_mask)
                hist += self.compute_batch_mIoU(y_batch_val, val_logits_2)
                # self.compute_mIoU(val_logits_2, y_batch_val)
                self.compute_val_acc(val_logits_2, y_batch_val)
                # self.test_gen_accuracy.update_state(y_batch_val, val_logits_2)
            val_iou_gen = self.compute_mIoU(hist)
            # val_class_acc = self.test_accuracy.result()
            val_class_gen_acc = self.test_gen_accuracy.result()
            # val_iou = self.validation_iou_metrics.result()
            # val_iou_gen = self.validation_iou_gen_metrics.result()

            #SAVING
            if val_iou_gen > self.acc_level:
                tf.saved_model.save(self.model, 'uda2_deeplab_aasp_model')
                self.acc_level = val_iou_gen
                print("Model Saved!")

            with self.train_summary_writer.as_default():
                tf.summary.scalar('pred2_val_acc', val_class_gen_acc, step=len(self.data_generator) * (epoch + 1))
                tf.summary.scalar('pred2_val_iou', val_iou_gen, step=len(self.data_generator) * (epoch + 1))

            # Save predicted images:
            # draw_labels(x_batch_val, y_batch_val, val_logits, self.batch_size, epoch, tr=False, gen=False)
            self.draw_labels(x_batch_val, y_batch_val, val_logits_2, self.batch_size, epoch, tr=False, gen=True)

            self.test_accuracy.reset_states()
            self.test_gen_accuracy.reset_states()
            self.validation_iou_metrics.reset_states()
            self.validation_iou_gen_metrics.reset_states()
            self.in_train_test_accuracy.reset_states()
            self.data_generator.on_epoch_end()
            self.validation_data_generator.on_epoch_end()
            self.in_train_test_accuracy.reset_states()
